import os

from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from monai.data import DataLoader
import argparse
from monai.utils import set_determinism
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler
from torch import autocast, nn
import time

from transformers import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup, \
    get_polynomial_decay_schedule_with_warmup

from training.augment import SingleLimitedLenWrapper, MultiLimitedLenWrapper


class Trainer:
    def __init__(self,
                 max_epochs,
                 batch_size,
                 num_step_per_epoch,
                 val_number,
                 device="cpu",
                 val_every=1,
                 logdir="./logs/",
                 is_single_process=False,
                 train_process=12,
                 weight_path=None
                 ):
        self.val_every = val_every
        self.max_epochs = max_epochs
        self.device = device
        self.batch_size = batch_size
        self.logdir = logdir
        self.scheduler = None
        self.model = None
        self.auto_optim = True
        self.warmup = 0.0
        self.scheduler_type = None
        self.weight_path = weight_path

        self.num_step_per_epoch = num_step_per_epoch
        self.val_number = val_number

        self.optimizer = None
        self.patch_size = None

        self.augmentation = True
        self.train_process = train_process
        self.print_time = False

        if self.device == "cpu":
            self.grad_scaler = None
        else:
            self.grad_scaler = GradScaler()

        torch.backends.cudnn.enabled = True


    def get_multi_processor_loader(self, train_ds, val_ds):
        from .augment import LimitedLenWrapper
        from .augment import get_train_transforms, get_validation_transforms, get_train_transforms_noaug, \
            get_train_transforms_nomirror, get_train_transforms_onlymirror, get_train_transforms_onlyspatial
        from datasets.dataloader import DataLoaderMultiProcess

        assert self.patch_size != None
        if self.augmentation:
            if self.augmentation == "nomirror":
                print(f"use augmentation: no mirror")
                tr_transforms = get_train_transforms_nomirror(patch_size=self.patch_size, mirror_axes=[0, 1, 2])
            elif self.augmentation == "onlymirror":
                print(f"use augmentation: only mirror")
                tr_transforms = get_train_transforms_onlymirror(patch_size=self.patch_size, mirror_axes=[0, 1, 2])
            elif self.augmentation == "onlyspatial":
                print(f"use augmentation: only spatial")
                tr_transforms = get_train_transforms_onlyspatial(patch_size=self.patch_size, mirror_axes=[0, 1, 2])

            else:
                tr_transforms = get_train_transforms(patch_size=self.patch_size, mirror_axes=[0, 1, 2])
        else:
            tr_transforms = get_train_transforms_noaug(patch_size=self.patch_size, mirror_axes=[0, 1, 2])

        val_transforms = get_validation_transforms()

        # train_loader = DataLoader(train_ds, num_workers=1, drop_last=True, shuffle=True, batch_size=self.batch_size)
        train_loader = DataLoaderMultiProcess(train_ds,
                                              batch_size=self.batch_size,
                                              patch_size=self.patch_size,
                                              print_time=self.print_time)
        # data_generator = MultiLimitedLenWrapper(self.num_step_per_epoch, data_loader=train_loader, num_processes=6,
        #                                    transform=tr_transforms, num_cached_per_queue=3, seeds=None,
        #                                    pin_memory=True, wait_time=0.02)
        data_generator = LimitedLenWrapper(self.num_step_per_epoch, data_loader=train_loader, num_processes=self.train_process,
                                           transform=tr_transforms, num_cached=8, seeds=None,
                                           pin_memory=True, wait_time=0.02)
        if val_ds is None:
            val_data_generator = None
        else:
            val_loader = DataLoaderMultiProcess(val_ds,
                                                batch_size=1,
                                                patch_size=self.patch_size,
                                                oversample_foreground_percent=1.0)
            # val_data_generator = MultiLimitedLenWrapper(self.val_number, data_loader=val_loader, transform=val_transforms,
            #                                        num_processes=6, num_cached_per_queue=2, seeds=None,
            #                                        pin_memory=True, wait_time=0.02)
            val_data_generator = LimitedLenWrapper(self.val_number, data_loader=val_loader, transform=val_transforms,
                                                   num_processes=6, num_cached=3, seeds=None,
                                                   pin_memory=True, wait_time=0.02)
        return data_generator, val_data_generator

    def get_single_processor_loader(self, train_ds, val_ds):
        from .augment import get_train_transforms, get_validation_transforms, get_train_transforms_noaug, \
            get_train_transforms_nomirror, get_train_transforms_onlymirror, get_train_transforms_onlyspatial

        assert self.patch_size != None
        if self.augmentation:
            if self.augmentation == "nomirror":
                print(f"use augmentation: no mirror")
                tr_transforms = get_train_transforms_nomirror(patch_size=self.patch_size, mirror_axes=[0, 1, 2])
            elif self.augmentation == "onlymirror":
                print(f"use augmentation: only mirror")
                tr_transforms = get_train_transforms_onlymirror(patch_size=self.patch_size, mirror_axes=[0, 1, 2])
            elif self.augmentation == "onlyspatial":
                print(f"use augmentation: only spatial")
                tr_transforms = get_train_transforms_onlyspatial(patch_size=self.patch_size, mirror_axes=[0, 1, 2])

            else:
                tr_transforms = get_train_transforms(patch_size=self.patch_size, mirror_axes=[0, 1, 2])
        else:
            tr_transforms = get_train_transforms_noaug(patch_size=self.patch_size, mirror_axes=[0, 1, 2])

        val_transforms = get_validation_transforms()

        train_loader = DataLoader(train_ds, num_workers=8, drop_last=True, shuffle=True, batch_size=self.batch_size)
        data_generator = SingleLimitedLenWrapper(self.num_step_per_epoch, data_loader=iter(train_loader), transform=tr_transforms)

        if val_ds is None:
            val_data_generator = None
        else:
            val_loader = DataLoader(val_ds, num_workers=8, drop_last=False, shuffle=False, batch_size=1)
            val_data_generator = SingleLimitedLenWrapper(self.val_number, data_loader=iter(val_loader), transform=val_transforms)
        return data_generator, val_data_generator

    def to_device(self, batch):
        if isinstance(batch, dict):
            for k, v in batch.items():
                if isinstance(batch[k], np.ndarray):
                    batch[k] = torch.from_numpy(batch[k])

                if (isinstance(batch[k], torch.Tensor) or isinstance(batch[k], torch.FloatTensor)):
                    batch[k] = batch[k].to(self.device).contiguous()

        elif isinstance(batch, list):
            batch = [torch.from_numpy(x) for x in batch if isinstance(x, np.ndarray)]
            batch = [x.to(self.device).contiguous() for x in batch if
                     (isinstance(x, torch.Tensor) or isinstance(x, torch.FloatTensor))]

        elif isinstance(batch, np.ndarray):
            batch = torch.from_numpy(batch)
            batch = batch.to(self.device).contiguous()

        else:
            print("not support data type")
            exit(0)

        return batch

    def validation_single_gpu(self, val_dataset):
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True)
        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()
        val_outputs = []

        for idx, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
            batch = self.before_data_to_device(batch)
            batch = self.to_device(batch)

            with torch.no_grad():
                val_out = self.validation_step(batch)
                assert val_out is not None

            return_list = False
            val_outputs.append(val_out)
        if isinstance(val_out, list) or isinstance(val_out, tuple):
            return_list = True

        val_outputs = torch.tensor(val_outputs)
        if not return_list:
            # 说明只有一个变量
            length = 0
            v_sum = 0.0
            for v in val_outputs:
                if not torch.isnan(v):
                    v_sum += v
                    length += 1

            if length == 0:
                v_sum = 0
            else:
                v_sum = v_sum / length
        else:
            num_val = len(val_outputs[0])
            length = [0.0 for i in range(num_val)]
            v_sum = [0.0 for i in range(num_val)]

            for v in val_outputs:
                for i in range(num_val):
                    if not torch.isnan(v[i]):
                        v_sum[i] += v[i]
                        length[i] += 1

            for i in range(num_val):
                if length[i] == 0:
                    v_sum[i] = 0
                else:
                    v_sum[i] = v_sum[i] / length[i]
        return v_sum, val_outputs

    def validate(self):
        val_outputs = []
        self.model.eval()


        outputs_split = None
        for i in tqdm(range(self.val_number), total=self.val_number):
            batch = next(self.val_loader)

            batch = self.before_data_to_device(batch)

            batch = self.to_device(batch)

            with torch.no_grad():
                with torch.autocast("cuda"):
                    val_out = self.validation_step(batch)
                    assert val_out is not None
                    if type(val_out) is not list and type(val_out) is not tuple:
                        val_out = [val_out]

                    if outputs_split is None:
                        outputs_split = [[] for i in range(len(val_out))]

                    for i, v in enumerate(val_out):
                        outputs_split[i].append(v)

            # val_outputs.append(val_out)


        val_outputs_merge = []
        for i in range(len(outputs_split)):
            val_outputs = torch.tensor(outputs_split[i])
            val_outputs_merge.append(val_outputs)
        # val_outputs = torch.tensor(val_outputs)

        if len(val_outputs_merge) == 1:
            val_outputs_merge = val_outputs_merge[0]
        self.validation_end(val_outputs_merge)

    def train(self,
              train_dataset,
              val_dataset=None,
              is_single_loader=False
              ):
        print(f"augmentation: {self.augmentation}")
        assert self.patch_size is not None, "please define the patch_size"

        if self.model is not None:
            print(
                f"check model parameter: {next(self.model.parameters()).sum()}, keep model parameters on different processes consistent")

        self.global_step = 0
        if self.model is not None:
            self.model.to(self.device)
        if self.weight_path:
            self.load_state_dict(self.weight_path)
        os.makedirs(self.logdir, exist_ok=True)
        self.writer = SummaryWriter(self.logdir)

        # self.train_loader, self.val_loader = self.get_multi_processor_loader(train_dataset, val_dataset)
        if is_single_loader:
            self.train_loader, self.val_loader = self.get_single_processor_loader(train_dataset, val_dataset)
        else:
            self.train_loader, self.val_loader = self.get_multi_processor_loader(train_dataset, val_dataset)

        self.max_steps = self.max_epochs * len(self.train_loader)

        print(f"step number is {self.max_steps}")

        if self.scheduler_type == "cosine_with_warmup":
            if self.warmup == 0.0:
                self.warmup = 0.1
            assert self.warmup < 1 and self.warmup > 0
            warmup_steps = self.max_steps * self.warmup
            self.scheduler = get_cosine_schedule_with_warmup(self.optimizer,
                                                             num_warmup_steps=warmup_steps,
                                                             num_training_steps=self.max_steps)
            print(f"warmup steps is {warmup_steps}")
        elif self.scheduler_type == "constant_with_warmup":
            if self.warmup == 0.0:
                self.warmup = 0.1
            assert self.warmup < 1 and self.warmup > 0
            warmup_steps = self.max_steps * self.warmup
            self.scheduler = get_constant_schedule_with_warmup(self.optimizer,
                                                               num_warmup_steps=warmup_steps,
                                                               )
            print(f"warmup steps is {warmup_steps}")

        elif self.scheduler_type == "poly_with_warmup":
            if self.warmup == 0.0:
                self.warmup = 0.1
            assert self.warmup < 1 and self.warmup > 0
            warmup_steps = self.max_steps * self.warmup
            self.scheduler = get_polynomial_decay_schedule_with_warmup(self.optimizer,
                                                                       num_warmup_steps=warmup_steps,
                                                                       num_training_steps=self.max_steps
                                                                       )
            print(f"warmup steps is {warmup_steps}")

        elif self.scheduler_type == "poly":
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            print(f"initial lr is {lr}")
            self.scheduler = PolyLRScheduler(self.optimizer, initial_lr=lr, max_steps=self.max_steps)
            print(f"scheduler_type is poly, warmup steps is {0}")

        for epoch in range(0, self.max_epochs):
            self.epoch = epoch
            self.train_epoch(epoch)
            if epoch == 0 or (epoch + 1) % self.val_every == 0:
                self.validate()

            if self.model is not None:
                self.model.train()

        if self.train_loader is not None:
            self.train_loader._finish()
        if self.val_loader is not None:
            self.val_loader._finish()


    def before_data_to_device(self, batch_data):
        return batch_data

    def train_epoch(self, epoch):
        if self.model is not None:
            self.model.train()

        with tqdm(total=self.num_step_per_epoch) as t:
            for i in range(self.num_step_per_epoch):
                self.global_step += 1
                t.set_description('Epoch %i' % epoch)

                if self.print_time:
                    s = time.time()
                batch = next(self.train_loader)
                if self.print_time:
                    e = time.time()
                    print(f"get batch time is {e - s}")

                batch = self.before_data_to_device(batch)

                batch = self.to_device(batch)

                if self.model is not None:
                    for param in self.model.parameters(): param.grad = None

                if not self.auto_optim:
                    loss = self.training_step(batch)
                else:
                    with autocast("cuda"):
                        if self.print_time:
                            s = time.time()
                        loss = self.training_step(batch)
                        if self.print_time:
                            e = time.time()
                            print(f"training step time is {e - s}")

                    if self.print_time:
                        s = time.time()

                    if self.grad_scaler is not None:
                        self.grad_scaler.scale(loss).backward()
                        self.grad_scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 12)
                        self.grad_scaler.step(self.optimizer)
                        self.grad_scaler.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 12)
                        self.optimizer.step()

                    if self.print_time:
                        e = time.time()
                        print(f"backward time is {e - s}")

                    if self.scheduler is not None:
                        self.scheduler.step()
                    lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                    self.log("lr", lr, self.global_step)

                    t.set_postfix(loss=loss.item(), lr=lr)

                t.update(1)

    def training_step(self, batch):
        raise NotImplementedError

    def validation_step(self, batch):
        raise NotImplementedError

    def validation_end(self, mean_val_outputs, val_outputs):
        pass

    def log(self, k, v, step):
        self.writer.add_scalar(k, scalar_value=v, global_step=step)

    def log_dict(self, dict_, step):
        for k, v in dict_.items():
            self.writer.add_scalar(k, scalar_value=v, global_step=step)

    def load_state_dict(self, weight_path, strict=True):
        sd = torch.load(weight_path, map_location=self.device)
        if "module" in sd:
            sd = sd["module"]
        new_sd = {}
        for k, v in sd.items():
            k = str(k)
            new_k = k[7:] if k.startswith("module") else k
            new_sd[new_k] = v

        self.model.load_state_dict(new_sd, strict=strict)

        print(f"model parameters are loaded successed.")

