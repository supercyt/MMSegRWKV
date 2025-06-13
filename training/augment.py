import numpy as np
import torch
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, \
    ContrastAugmentationTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, NumpyToTensor

from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter

# Copyright 2021 Division of Medical Image Computing, German Cancer Research Center (DKFZ)
# and Applied Computer Vision Lab, Helmholtz Imaging Platform
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

class SingleThreadedAugmenter(object):
    """
    Use this for debugging custom transforms. It does not use a background thread and you can therefore easily debug
    into your augmentations. This should not be used for training. If you want a generator that uses (a) background
    process(es), use MultiThreadedAugmenter.
    Args:
        data_loader (generator or DataLoaderBase instance): Your data loader. Must have a .next() function and return
        a dict that complies with our data structure

        transform (Transform instance): Any of our transformations. If you want to use multiple transformations then
        use our Compose transform! Can be None (in that case no transform will be applied)
    """
    def __init__(self, data_loader, transform):
        self.data_loader = data_loader
        self.transform = transform

    def __iter__(self):
        return self

    def __next__(self):
        item = next(self.data_loader)
        if self.transform is not None:
            for k, v in item.items():
                if type(v) is torch.Tensor:
                    item[k] = v.cpu().numpy()
            item = self.transform(**item)
        return item

    def next(self):
        return self.__next__()


class LimitedLenWrapper(NonDetMultiThreadedAugmenter):
    def __init__(self, my_imaginary_length, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.len = my_imaginary_length

    def __len__(self):
        return self.len


class MultiLimitedLenWrapper(MultiThreadedAugmenter):
    def __init__(self, my_imaginary_length, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.len = my_imaginary_length

    def __len__(self):
        return self.len

class SingleLimitedLenWrapper(SingleThreadedAugmenter):
    def __init__(self, my_imaginary_length, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.len = my_imaginary_length

    def __len__(self):
        return self.len


def get_train_transforms(patch_size, mirror_axes=None):
    tr_transforms = []
    patch_size_spatial = patch_size
    ignore_axes = None
    angle = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)

    tr_transforms.append(SpatialTransform(
        patch_size_spatial, patch_center_dist_from_border=None,
        do_elastic_deform=False, alpha=(0, 0), sigma=(0, 0),
        do_rotation=True, angle_x=angle, angle_y=angle, angle_z=angle,
        p_rot_per_axis=1,  # todo experiment with this
        do_scale=True, scale=(0.7, 1.4),
        border_mode_data="constant", border_cval_data=0, order_data=3,
        border_mode_seg="constant", border_cval_seg=-1, order_seg=1,
        random_crop=False,  # random cropping is part of our dataloaders
        p_el_per_sample=0, p_scale_per_sample=0.2, p_rot_per_sample=0.2,
        independent_scale_for_each_axis=False  # todo experiment with this
    ))

    tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))
    tr_transforms.append(GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2,
                                               p_per_channel=0.5))
    tr_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15))
    tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))
    tr_transforms.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
                                                        p_per_channel=0.5,
                                                        order_downsample=0, order_upsample=3, p_per_sample=0.25,
                                                        ignore_axes=ignore_axes))
    tr_transforms.append(GammaTransform((0.7, 1.5), True, True, retain_stats=True, p_per_sample=0.1))
    tr_transforms.append(GammaTransform((0.7, 1.5), False, True, retain_stats=True, p_per_sample=0.3))

    if mirror_axes is not None and len(mirror_axes) > 0:
        tr_transforms.append(MirrorTransform(mirror_axes))

    tr_transforms.append(RemoveLabelTransform(-1, 0))
    tr_transforms.append(NumpyToTensor(['data', 'seg'], 'float'))

    tr_transforms = Compose(tr_transforms)

    return tr_transforms


def get_train_transforms_nomirror(patch_size, mirror_axes=None):
    tr_transforms = []
    patch_size_spatial = patch_size
    ignore_axes = None
    angle = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)

    tr_transforms.append(SpatialTransform(
        patch_size_spatial, patch_center_dist_from_border=None,
        do_elastic_deform=False, alpha=(0, 0), sigma=(0, 0),
        do_rotation=True, angle_x=angle, angle_y=angle, angle_z=angle,
        p_rot_per_axis=1,  # todo experiment with this
        do_scale=True, scale=(0.7, 1.4),
        border_mode_data="constant", border_cval_data=0, order_data=3,
        border_mode_seg="constant", border_cval_seg=-1, order_seg=1,
        random_crop=False,  # random cropping is part of our dataloaders
        p_el_per_sample=0, p_scale_per_sample=0.2, p_rot_per_sample=0.2,
        independent_scale_for_each_axis=False  # todo experiment with this
    ))

    tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))
    tr_transforms.append(GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2,
                                               p_per_channel=0.5))
    tr_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15))
    tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))
    tr_transforms.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
                                                        p_per_channel=0.5,
                                                        order_downsample=0, order_upsample=3, p_per_sample=0.25,
                                                        ignore_axes=ignore_axes))
    tr_transforms.append(GammaTransform((0.7, 1.5), True, True, retain_stats=True, p_per_sample=0.1))
    tr_transforms.append(GammaTransform((0.7, 1.5), False, True, retain_stats=True, p_per_sample=0.3))

    # if mirror_axes is not None and len(mirror_axes) > 0:
    #     tr_transforms.append(MirrorTransform(mirror_axes))

    tr_transforms.append(RemoveLabelTransform(-1, 0))
    tr_transforms.append(NumpyToTensor(['data', 'seg'], 'float'))

    tr_transforms = Compose(tr_transforms)

    return tr_transforms


def get_train_transforms_onlymirror(patch_size, mirror_axes=None):
    tr_transforms = []
    patch_size_spatial = patch_size
    ignore_axes = None
    angle = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)

    tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))
    tr_transforms.append(GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2,
                                               p_per_channel=0.5))
    tr_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15))
    tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))
    tr_transforms.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
                                                        p_per_channel=0.5,
                                                        order_downsample=0, order_upsample=3, p_per_sample=0.25,
                                                        ignore_axes=ignore_axes))
    tr_transforms.append(GammaTransform((0.7, 1.5), True, True, retain_stats=True, p_per_sample=0.1))
    tr_transforms.append(GammaTransform((0.7, 1.5), False, True, retain_stats=True, p_per_sample=0.3))

    if mirror_axes is not None and len(mirror_axes) > 0:
        tr_transforms.append(MirrorTransform(mirror_axes))

    tr_transforms.append(RemoveLabelTransform(-1, 0))
    tr_transforms.append(NumpyToTensor(['data', 'seg'], 'float'))

    tr_transforms = Compose(tr_transforms)

    return tr_transforms


def get_train_transforms_onlyspatial(patch_size, mirror_axes=None):
    tr_transforms = []
    patch_size_spatial = patch_size
    ignore_axes = None
    angle = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)

    tr_transforms.append(SpatialTransform(
        patch_size_spatial, patch_center_dist_from_border=None,
        do_elastic_deform=False, alpha=(0, 0), sigma=(0, 0),
        do_rotation=True, angle_x=angle, angle_y=angle, angle_z=angle,
        p_rot_per_axis=1,  # todo experiment with this
        do_scale=True, scale=(0.7, 1.4),
        border_mode_data="constant", border_cval_data=0, order_data=3,
        border_mode_seg="constant", border_cval_seg=-1, order_seg=1,
        random_crop=False,  # random cropping is part of our dataloaders
        p_el_per_sample=0, p_scale_per_sample=0.2, p_rot_per_sample=0.2,
        independent_scale_for_each_axis=False  # todo experiment with this
    ))

    if mirror_axes is not None and len(mirror_axes) > 0:
        tr_transforms.append(MirrorTransform(mirror_axes))

    tr_transforms.append(RemoveLabelTransform(-1, 0))
    tr_transforms.append(NumpyToTensor(['data', 'seg'], 'float'))

    tr_transforms = Compose(tr_transforms)

    return tr_transforms


def get_train_transforms_noaug(patch_size, mirror_axes=None):
    tr_transforms = []

    tr_transforms.append(RemoveLabelTransform(-1, 0))
    tr_transforms.append(NumpyToTensor(['data', 'seg'], 'float'))

    tr_transforms = Compose(tr_transforms)

    return tr_transforms


def get_validation_transforms() -> AbstractTransform:
    val_transforms = []
    val_transforms.append(RemoveLabelTransform(-1, 0))

    val_transforms.append(NumpyToTensor(['data', 'seg'], 'float'))
    val_transforms = Compose(val_transforms)
    return val_transforms
