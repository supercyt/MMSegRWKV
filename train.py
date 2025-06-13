import numpy as np

import torch
import torch.nn as nn
from monai.inferers import SlidingWindowInferer
from monai.losses import DiceCELoss
from monai.utils import set_determinism

from datasets.brats import get_brats_loader_from_train, get_single_process_loader_from_train
from evaluation.metric import dice
from monai.networks.nets import UNETR, SwinUNETR, SegResNet
from models.backbones.NNFormer.nnFormer_tumor import nnFormer
from models.backbones.MedNetXt.mednextv1.create_mednext_v1 import create_mednext_v1

from training.trainer import Trainer
from training.utils import save_new_model_and_delete_last

set_determinism(123)
import os

architecture = "mmsegrwkv"

# logdir = f"./logs/segrwkv-brats2023"
logdir = f"./logs/segrwkv-isles2022"
# logdir = f"./logs/segrwkv-prostate"

model_save_path = os.path.join(logdir, architecture)
# augmentation = "nomirror"
augmentation = True
# roi_size = [128, 128, 128]
roi_size = [64, 128, 128]
# roi_size = [64, 64, 64]

DATA_TYPE = "iseles"

# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.allow_tf32 = False
# torch.backends.cuda.matmul.allow_tf32 = False

def func(m, epochs):
    return np.exp(-10 * (1 - m / epochs) ** 2)


class BraTSTrainer(Trainer):
    def __init__(self, model, max_epochs, batch_size, num_step_per_epoch, val_number, device="cpu",
                 val_every=1, logdir="./logs/", weight_path=None):
        super().__init__(max_epochs, batch_size, num_step_per_epoch, val_number, device, val_every, weight_path=weight_path)
        self.window_infer = SlidingWindowInferer(roi_size=roi_size, sw_batch_size=1, overlap=0.5)
        self.augmentation = augmentation
        self.model = model
        self.patch_size = roi_size
        self.best_mean_dice = 0.0
        self.train_process = 18
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=1e-2, weight_decay=1e-6, momentum=0.99, nesterov=True)

        self.scheduler_type = "poly"
        self.cross = nn.CrossEntropyLoss()
        # self.dice_loss = DiceCELoss(to_onehot_y=True, lambda_ce=0.5, lambda_dice=0.5)

    def training_step(self, batch):
        image, label = self.get_input(batch)

        pred = self.model(image)

        loss = self.cross(pred, label)
        # loss = self.dice_loss(pred, label.unsqueeze(1))

        self.log("training_loss", loss, step=self.global_step)

        return loss

    def convert_labels(self, labels):
        ## TC, WT and ET
        result = [(labels == 1) | (labels == 3), (labels == 1) | (labels == 3) | (labels == 2), labels == 3]

        return torch.cat(result, dim=1).float()

    def convert_prostate_labels(self, labels):
        ## TC, WT and ET
        result = [labels == 1,  labels == 2]

        return torch.cat(result, dim=1).float()

    def get_input(self, batch):
        image = batch["data"]
        label = batch["seg"]

        label = label[:, 0].long()
        return image, label

    def cal_metric(self, gt, pred, voxel_spacing=(1.0, 1.0, 1.0)):
        d = dice(pred, gt, nan_for_nonexisting=False)
        return np.array([d, 50])

    def validation_step(self, batch):
        image, label = self.get_input(batch)

        output = self.model(image)

        output = output.argmax(dim=1)
        output = output[:, None]
        label = label[:, None]
        if "brats" in DATA_TYPE.lower():
            output = self.convert_labels(output)
            label = self.convert_labels(label)
            c = 3
        elif "isels" in DATA_TYPE.lower():
            c = 1
        elif "prostate" in DATA_TYPE.lower():
            output = self.convert_prostate_labels(output)
            label = self.convert_prostate_labels(label)
            c = 2

        output = output.cpu().numpy()
        target = label.cpu().numpy()

        dices = []

        for i in range(0, c):
            pred_c = output[:, i]
            target_c = target[:, i]

            cal_dice, _ = self.cal_metric(target_c, pred_c)
            dices.append(cal_dice)

        return dices

    def validation_end(self, val_outputs):
        dices = val_outputs
        if "brats" in DATA_TYPE.lower():
            tc, wt, et = dices[0].mean(), dices[1].mean(), dices[2].mean()

            print(f"dices is {tc, wt, et}")

            mean_dice = (tc + wt + et) / 3

            self.log("tc", tc, step=self.epoch)
            self.log("wt", wt, step=self.epoch)
            self.log("et", et, step=self.epoch)

            self.log("mean_dice", mean_dice, step=self.epoch)
        elif "isels" in DATA_TYPE.lower():
            mean_dice = dices[0].mean()
            self.log("mean_dice", mean_dice, step=self.epoch)
        elif "prostate" in  DATA_TYPE.lower():
            pz, tz = dices[0].mean(), dices[1].mean()

            print(f"dices is {pz, tz}")

            mean_dice = (pz + tz) / 2

            self.log("pz", pz, step=self.epoch)
            self.log("tz", tz, step=self.epoch)

            self.log("mean_dice", mean_dice, step=self.epoch)

        if mean_dice > self.best_mean_dice:
            self.best_mean_dice = mean_dice
            save_new_model_and_delete_last(self.model,
                                           os.path.join(model_save_path,
                                                        f"best_model_{mean_dice:.4f}.pt"),
                                           delete_symbol="best_model")

        save_new_model_and_delete_last(self.model,
                                       os.path.join(model_save_path,
                                                    f"final_model_{mean_dice:.4f}.pt"),
                                       delete_symbol="final_model")

        if (self.epoch + 1) % 50 == 0:
            torch.save(self.model.state_dict(),
                       os.path.join(model_save_path, f"tmp_model_ep{self.epoch}_{mean_dice:.4f}.pt"))

        print(f"mean_dice is {mean_dice}")


if __name__ == "__main__":
    max_epoch = 2000
    batch_size = 2
    val_every = 2
    device = "cuda:0"
    is_single_loader = False
    weight_path = None
    if "brats" in DATA_TYPE.lower():
        data_dir = f"/home/caoyitong/DataProjects/brats2023/data/fullres/train"
        in_chans = 4
        out_chans = 4
        spatial_size = 3
        seed = 42
        num_step_per_epoch = 250
        val_number = 100
    elif "isles" in DATA_TYPE.lower():
        data_dir = f"/home/caoyitong/DataProjects/ISLES-2022/data/fullres/train"
        in_chans = 3
        out_chans = 2
        spatial_size = 3
        seed = 130
        num_step_per_epoch = 250
        val_number = 100
    elif "prostate" in DATA_TYPE.lower():
        data_dir = f"/home/caoyitong/DataProjects/prostate/data/fullres/train"
        in_chans = 2
        out_chans = 3
        spatial_size = 3
        seed = 7330479
        num_step_per_epoch = 100
        val_number = 50

    if architecture == "mmsegrwkv":
        from models.backbones.MMSegRWKV.mmsegrwkv import MMSegRWKV
        model = MMSegRWKV(in_chans=in_chans, out_chans=out_chans,
                            depths=[2, 2, 2, 2], feat_size=[32, 64, 128, 256], window_sizes=[8, 8, 4, 4])
    # SegResNet
    elif architecture == "segres_net":
        model = SegResNet(spatial_dims=spatial_size,
                          init_filters=32,
                          in_channels=in_chans,
                          out_channels=out_chans,
                          dropout_prob=0.2,
                          blocks_down=(1, 2, 2, 4),
                          blocks_up=(1, 1, 1)).to(device)
    # UNETR
    elif architecture == "unet_r":
        model = UNETR(in_channels=in_chans,
                      out_channels=out_chans,
                      img_size=roi_size,
                      proj_type='conv',
                      norm_name='instance')
    # SwinUNETR
    elif architecture == "swinunet_r":
        model = SwinUNETR(
            img_size=roi_size,
            in_channels=in_chans,
            out_channels=out_chans,
            feature_size=48,
            drop_rate=0.1,
            attn_drop_rate=0.2,
            dropout_path_rate=0.1,
            spatial_dims=spatial_size,
            use_checkpoint=False,
            use_v2=False)
    # SwinUNETR
    elif architecture == "swinunet_r_v2":
        model = SwinUNETR(
            img_size=roi_size,
            in_channels=in_chans,
            out_channels=out_chans,
            feature_size=48,
            drop_rate=0.1,
            attn_drop_rate=0.2,
            dropout_path_rate=0.1,
            spatial_dims=spatial_size,
            use_checkpoint=False,
            use_v2=True)
    # nnFormer
    elif architecture == "nn_former":
        model = nnFormer(crop_size=np.array(roi_size),
                         embedding_dim=96,
                         input_channels=in_chans,
                         num_classes=out_chans,
                         depths=[2, 2, 2, 2],
                         num_heads=[3, 6, 12, 24],
                         deep_supervision=False,
                         conv_op=nn.Conv3d,
                         patch_size=[4, 4, 4],
                         window_size=[4, 4, 8, 4])
    # nnFormer
    elif architecture == "mednext":
        model = create_mednext_v1(
            num_input_channels=in_chans,
            num_classes=out_chans,
            model_id='S',  # S, B, M and L are valid model ids
            kernel_size=3,  # 3x3x3 and 5x5x5 were tested in publication
            deep_supervision=True  # was used in publication
        )
    if is_single_loader:
        train_ds, val_ds, test_ds = get_single_process_loader_from_train(
            data_dir, patch_size=roi_size, seed=seed)
    else:
        train_ds, val_ds, test_ds = get_brats_loader_from_train(data_dir, seed=seed)

    # num_step_per_epoch = math.ceil(len(train_ds) // batch_size)
    # # BraTS23
    # num_step_per_epoch = 500
    # val_number = 150
    # AIIB23

    trainer = BraTSTrainer(
        model=model,
        max_epochs=max_epoch,
        batch_size=batch_size,
        num_step_per_epoch=num_step_per_epoch,
        val_number=val_number,
        device=device,
        logdir=logdir,
        val_every=val_every,
        weight_path=weight_path
    )

    trainer.train(train_dataset=train_ds, val_dataset=test_ds, is_single_loader=is_single_loader)