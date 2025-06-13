import os

import numpy as np
import torch
from monai.inferers import SlidingWindowInferer
from torch import nn

from datasets.brats import get_brats_loader_from_train
from monai.networks.nets import UNETR, SwinUNETR, SegResNet
from models.backbones.NNFormer.nnFormer_tumor import nnFormer
from models.backbones.MedNetXt.mednextv1.create_mednext_v1 import create_mednext_v1

from training.predictor import Predictor
from training.trainer import Trainer

roi_size = [32, 128, 128]
DATA_TYPE = "inhouse"


class BraTSPrector(Trainer):
    def __init__(self,architecture, max_epochs, batch_size, in_chans, out_chans, model_path, save_path,
                 num_step_per_epoch=0, val_number=0, device="cpu", val_every=1,logdir="./logs/"):
        super().__init__(max_epochs, batch_size, num_step_per_epoch, val_number, device, val_every, logdir)

        self.patch_size = roi_size
        self.augmentation = False

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.model_path = model_path
        self.save_path = save_path
        self.architecture = architecture

    def convert_labels(self, labels):
        ## TC, WT and ET
        result = [(labels == 1) | (labels == 3), (labels == 1) | (labels == 3) | (labels == 2), labels == 3]

        return torch.cat(result, dim=1).float()

    def get_input(self, batch):
        image = batch["data"]
        label = batch["seg"]
        properties = batch["properties"]
        # label = self.convert_labels(label)

        return image, label, properties

    def define_model_segrwkv(self):
        if self.architecture == "mmsegrwkv":
            from models.backbones.MMSegRWKV.mmsegrwkv import MMSegRWKV
            # from models.backbones.MMSegRWKV.mmsegrwkv_l import MMSegRWKV
            model = MMSegRWKV(in_chans=self.in_chans, out_chans=self.out_chans, 
                              depths=[2, 2, 2, 2], feat_size=[32, 64, 128, 256], window_sizes=[8, 8, 4, 4])
        # SegResNet
        elif self.architecture == "segres_net":
            model = SegResNet(spatial_dims=3,
                              init_filters=32,
                              in_channels=self.in_chans,
                              out_channels=self.out_chans,
                              dropout_prob=0.2,
                              blocks_down=(1, 2, 2, 4),
                              blocks_up=(1, 1, 1))
        # UNETR
        elif self.architecture == "unet_r":
            model = UNETR(in_channels=self.in_chans,
                          out_channels=self.out_chans,
                          img_size=roi_size,
                          proj_type='conv',
                          norm_name='instance')
        # SwinUNETR
        elif self.architecture == "swinunet_r":
            model = SwinUNETR(
                img_size=roi_size,
                in_channels=self.in_chans,
                out_channels=self.out_chans,
                feature_size=48,
                drop_rate=0.1,
                attn_drop_rate=0.2,
                dropout_path_rate=0.1,
                spatial_dims=3,
                use_checkpoint=False,
                use_v2=False)
        # SwinUNETR
        elif self.architecture == "swinunet_r_v2":
            model = SwinUNETR(
                img_size=roi_size,
                in_channels=self.in_chans,
                out_channels=self.out_chans,
                feature_size=48,
                drop_rate=0.1,
                attn_drop_rate=0.2,
                dropout_path_rate=0.1,
                spatial_dims=3,
                use_checkpoint=False,
                use_v2=True)
        # nnFormer
        elif self.architecture == "nn_former":
            model = nnFormer(crop_size=np.array(roi_size),
                             embedding_dim=96,
                             input_channels=self.in_chans,
                             num_classes=self.out_chans,
                             depths=[2, 2, 2, 2],
                             num_heads=[3, 6, 12, 24],
                             deep_supervision=False,
                             conv_op=nn.Conv3d,
                             patch_size=[4, 4, 4],
                             window_size=[4, 4, 8, 4])
        # nnFormer
        elif self.architecture == "mednext":
            model = create_mednext_v1(
                num_input_channels=self.in_chans,
                num_classes=self.out_chans,
                model_id='S',  # S, B, M and L are valid model ids
                kernel_size=3  # 3x3x3 and 5x5x5 were tested in publication
            )
        new_sd = self.filte_state_dict(torch.load(self.model_path, map_location=self.device))
        model.load_state_dict(new_sd)
        model.eval()
        window_infer = SlidingWindowInferer(roi_size=roi_size,
                                            sw_batch_size=2,
                                            overlap=0.5,
                                            progress=True,
                                            mode="gaussian")

        predictor = Predictor(window_infer=window_infer, mirror_axes=[0, 1, 2])

        os.makedirs(self.save_path, exist_ok=True)

        return model, predictor, self.save_path

    def validation_step(self, batch):
        image, label, properties = self.get_input(batch)
        ddim = False
        print(properties["name"])
        model, predictor, save_path = self.define_model_segrwkv()

        model_output = predictor.maybe_mirror_and_predict(image, model, device=self.device)

        model_output = predictor.predict_raw_probability(model_output, properties=properties)
        model_output = model_output.argmax(dim=0)[None]

        if "brats" in DATA_TYPE.lower():
            model_output = self.convert_labels_dim0(model_output)
        elif "prostate" in DATA_TYPE.lower():
            model_output = self.convert_prostate_labels_dim0(model_output)

        model_output = predictor.predict_noncrop_probability(model_output, properties)
        predictor.save_to_nii(model_output,
                              raw_spacing=[1, 1, 1],
                              case_name=properties['name'][0],
                              save_dir=save_path)

        return 0

    def convert_labels_dim0(self, labels):
        ## TC, WT and ET
        result = [(labels == 1) | (labels == 3), (labels == 1) | (labels == 3) | (labels == 2), labels == 3]

        return torch.cat(result, dim=0).float()

    def convert_prostate_labels_dim0(self, labels):
        ## TC, WT and ET
        result = [labels == 1, labels == 2]

        return torch.cat(result, dim=0).float()

    def filte_state_dict(self, sd):
        if "module" in sd:
            sd = sd["module"]
        new_sd = {}
        for k, v in sd.items():
            k = str(k)
            new_k = k[7:] if k.startswith("module") else k
            new_sd[new_k] = v
        del sd
        return new_sd


if __name__ == "__main__":
    augmentation = True

    max_epoch = 600
    batch_size = 2
    val_every = 10
    num_gpus = 1
    device = "cuda:0"
    roi_size = [32, 128, 128]

    if "brats" in DATA_TYPE.lower():
        data_dir = f""
        model_path = ""
        save_path = "./prediction_results/segrwkv-brats"
        in_chans = 4
        out_chans = 4
        seed = 42
    elif "aiib" in DATA_TYPE.lower():
        data_dir = f""
        model_path = ""
        save_path = "./prediction_results/segrwkv-aiib"
        in_chans = 1
        out_chans = 2
        seed = 42
    elif "prostate" in DATA_TYPE.lower():
        data_dir = f""
        model_path = ""
        save_path = ""
        in_chans = 2
        out_chans = 3
        seed = 42

    validator = BraTSPrector(
        max_epochs=max_epoch,
        batch_size=batch_size,
        in_chans=in_chans,
        out_chans=out_chans,
        model_path=model_path,
        save_path=save_path,
        device=device,
        logdir="",
        val_every=val_every)

    train_ds, val_ds, test_ds = get_brats_loader_from_train(data_dir, seed=seed)
    validator.validation_single_gpu(test_ds)
