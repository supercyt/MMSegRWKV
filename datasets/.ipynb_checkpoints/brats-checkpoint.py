
# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import glob
import os
import pickle
import random
from typing import Union, Tuple

import numpy as np
from sklearn.model_selection import KFold  ## K折交叉验证
from torch.utils.data import Dataset
from tqdm import tqdm
from batchgenerators.utilities.file_and_folder_operations import isfile, subfiles
import multiprocessing


def _convert_to_npy(npz_file: str, unpack_segmentation: bool = True, overwrite_existing: bool = False) -> None:
    # try:
    a = np.load(npz_file)  # inexpensive, no compression is done here. This just reads metadata
    if overwrite_existing or not isfile(npz_file[:-3] + "npy"):
        np.save(npz_file[:-3] + "npy", a['data'])

    if unpack_segmentation and (overwrite_existing or not isfile(npz_file[:-4] + "_seg.npy")):
        np.save(npz_file[:-4] + "_seg.npy", a['seg'])



def unpack_dataset(folder: str, unpack_segmentation: bool = True, overwrite_existing: bool = False,
                   num_processes: int = 8):
    """
    all npz files in this folder belong to the dataset, unpack them all
    """
    # with multiprocessing.get_context("spawn").Pool(num_processes) as p:
    #     npz_files = subfiles(folder, True, None, ".npz", True)
    #     p.starmap(_convert_to_npy, zip(npz_files,
    #                                    [unpack_segmentation] * len(npz_files),
    #                                    [overwrite_existing] * len(npz_files))
    #               )
    npz_files = subfiles(folder, True, None, ".npz", True)
    return [_convert_to_npy(npz_file, unpack_segmentation, overwrite_existing) for npz_file in npz_files]



class BraTSDataset(Dataset):
    def __init__(self, datalist, is_test=False, need_unpacking=True) -> None:
        super().__init__()

        self.datalist = datalist
        self.is_test = is_test

        self.data_cached = []
        for p in tqdm(self.datalist, total=len(self.datalist)):
            info = self.load_pkl(p)

            self.data_cached.append(info)

        ## unpacking
        print(f"unpacking data ....")
        # for 
        folder = []
        for p in self.datalist:
            f = os.path.dirname(p)
            if f not in folder:
                folder.append(f)
        if need_unpacking:
            for f in folder:
                unpack_dataset(f, unpack_segmentation=True, overwrite_existing=False, num_processes=8)


        print(f"data length is {len(self.datalist)}")

    def load_pkl(self, data_path):
        pass
        properties_path = f"{data_path[:-4]}.pkl"
        df = open(properties_path, "rb")
        info = pickle.load(df)

        return info

    def post(self, batch_data):
        return batch_data

    def read_data(self, data_path):

        image_path = data_path.replace(".npz", ".npy")
        seg_path = data_path.replace(".npz", "_seg.npy")
        image_data = np.load(image_path, "r+")

        seg_data = None
        if not self.is_test:
            seg_data = np.load(seg_path, "r+")

        return image_data, seg_data

    def __getitem__(self, i):
        image, seg = self.read_data(self.datalist[i])

        properties = self.data_cached[i]

        if seg is None:
            return {
                "data": image,
                "properties": properties
            }
        else :
            return {
                "data": image,
                "seg": seg,
                "properties": properties
            }

    def __len__(self):
        return len(self.datalist)


class SingleProcessDataset(Dataset):
    def __init__(self, datalist, patch_size, is_test=False) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.datalist = datalist
        self.is_test = is_test
        self.need_to_pad = (np.array([0, 0, 0])).astype(int)

        self.data_cached = []
        for p in tqdm(self.datalist, total=len(self.datalist)):
            info = self.load_pkl(p)

            self.data_cached.append(info)

        ## unpacking
        print(f"unpacking data ....")
        # for
        folder = []
        for p in self.datalist:
            f = os.path.dirname(p)
            if f not in folder:
                folder.append(f)

        for f in folder:
            unpack_dataset(f, unpack_segmentation=True, overwrite_existing=False, num_processes=8)


        print(f"data length is {len(self.datalist)}")

    def load_pkl(self, data_path):
        pass
        properties_path = f"{data_path[:-4]}.pkl"
        df = open(properties_path, "rb")
        info = pickle.load(df)

        return info

    def post(self, batch_data):
        return batch_data

    def read_data(self, data_path):

        image_path = data_path.replace(".npz", ".npy")
        seg_path = data_path.replace(".npz", "_seg.npy")
        image_data = np.load(image_path, "r+")

        seg_data = None
        if not self.is_test:
            seg_data = np.load(seg_path, "r+")

        return image_data, seg_data

    def __getitem__(self, i):
        data, seg = self.read_data(self.datalist[i])
        properties = self.data_cached[i]
        case_properties = []

        # force_fg = self._oversample_last_XX_percent(j)
        force_fg = False
        case_properties.append(properties)
        # If we are doing the cascade then the segmentation from the previous stage will already have been loaded by
        # self._data.load_case(i) (see nnUNetDataset.load_case)
        shape = data.shape[1:]
        dim = len(shape)

        bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])
        # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
        # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
        # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
        # later
        valid_bbox_lbs = [max(0, bbox_lbs[i]) for i in range(dim)]
        valid_bbox_ubs = [min(shape[i], bbox_ubs[i]) for i in range(dim)]

        # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
        # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
        # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
        # remove label -1 in the data augmentation but this way it is less error prone)
        this_slice = tuple(
            [slice(0, data.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
        data = data[this_slice]

        this_slice = tuple([slice(0, seg.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
        seg = seg[this_slice]

        padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
        image = np.pad(data, ((0, 0), *padding), 'constant', constant_values=0)
        if seg is not None:
            seg = np.pad(seg, ((0, 0), *padding), 'constant', constant_values=0)

        if seg is None:
            return {
                "data": image,
                "properties": properties
            }
        else :
            return {
                "data": image,
                "seg": seg,
                "properties": properties
            }

    def __len__(self):
        return len(self.datalist)

    def _oversample_last_XX_percent(self, sample_idx: int) -> bool:
        """
        determines whether sample sample_idx in a minibatch needs to be guaranteed foreground
        """
        return not sample_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))

    def _probabilistic_oversampling(self, sample_idx: int) -> bool:
        # print('YEAH BOIIIIII')
        return np.random.uniform() < self.oversample_foreground_percent

    def get_bbox(self, data_shape: np.ndarray, force_fg: bool, class_locations: Union[dict, None],
                 overwrite_class: Union[int, Tuple[int, ...]] = None, verbose: bool = False):
        # in dataloader 2d we need to select the slice prior to this and also modify the class_locations to only have
        # locations for the given slice
        need_to_pad = self.need_to_pad.copy()
        dim = len(data_shape)

        for d in range(dim):
            # if case_all_data.shape + need_to_pad is still < patch size we need to pad more! We pad on both sides
            # always
            if need_to_pad[d] + data_shape[d] < self.patch_size[d]:
                need_to_pad[d] = self.patch_size[d] - data_shape[d]

        # we can now choose the bbox from -need_to_pad // 2 to shape - patch_size + need_to_pad // 2. Here we
        # define what the upper and lower bound can be to then sample form them with np.random.randint
        lbs = [- need_to_pad[i] // 2 for i in range(dim)]
        ubs = [data_shape[i] + need_to_pad[i] // 2 + need_to_pad[i] % 2 - self.patch_size[i] for i in range(dim)]

        # if not force_fg then we can just sample the bbox randomly from lb and ub. Else we need to make sure we get
        # at least one of the foreground classes in the patch
        if not force_fg:
            bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]
            # print('I want a random location')
        else:
            assert class_locations is not None, 'if force_fg is set class_locations cannot be None'
            if overwrite_class is not None:
                assert overwrite_class in class_locations.keys(), 'desired class ("overwrite_class") does not ' \
                                                                  'have class_locations (missing key)'
            # this saves us a np.unique. Preprocessing already did that for all cases. Neat.
            # class_locations keys can also be tuple
            eligible_classes_or_regions = [i for i in class_locations.keys() if len(class_locations[i]) > 0]

            # if we have annotated_classes_key locations and other classes are present, remove the annotated_classes_key from the list
            # strange formulation needed to circumvent
            # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
            # tmp = [i == self.annotated_classes_key if isinstance(i, tuple) else False for i in eligible_classes_or_regions]
            # if any(tmp):
            #     if len(eligible_classes_or_regions) > 1:
            #         eligible_classes_or_regions.pop(np.where(tmp)[0][0])

            if len(eligible_classes_or_regions) == 0:
                # this only happens if some image does not contain foreground voxels at all
                selected_class = None
                if verbose:
                    print('case does not contain any foreground classes')
            else:
                # I hate myself. Future me aint gonna be happy to read this
                # 2022_11_25: had to read it today. Wasn't too bad
                selected_class = eligible_classes_or_regions[np.random.choice(len(eligible_classes_or_regions))] if \
                    (overwrite_class is None or (
                                overwrite_class not in eligible_classes_or_regions)) else overwrite_class
            # print(f'I want to have foreground, selected class: {selected_class}')

            voxels_of_that_class = class_locations[selected_class] if selected_class is not None else None

            if voxels_of_that_class is not None and len(voxels_of_that_class) > 0:
                selected_voxel = voxels_of_that_class[np.random.choice(len(voxels_of_that_class))]
                # selected voxel is center voxel. Subtract half the patch size to get lower bbox voxel.
                # Make sure it is within the bounds of lb and ub
                # i + 1 because we have first dimension 0!
                bbox_lbs = [max(lbs[i], selected_voxel[i + 1] - self.patch_size[i] // 2) for i in range(dim)]
            else:
                # If the image does not contain any foreground classes, we fall back to random cropping
                bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]

        bbox_ubs = [bbox_lbs[i] + self.patch_size[i] for i in range(dim)]

        return bbox_lbs, bbox_ubs

def get_brats_loader_from_train(data_dir, train_rate=0.7, val_rate=0.1, test_rate=0.2, seed=42):
    ## training all labeled data
    ## fold denote the validation data in training data
    all_paths = glob.glob(f"{data_dir}/*.npz")

    train_number = int(len(all_paths) * train_rate)
    val_number = int(len(all_paths) * val_rate)
    # test_number = int(len(all_paths) * test_rate)
    test_number = len(all_paths) - train_number - val_number
    random.seed(seed)
    random.shuffle(all_paths)

    train_datalist = all_paths[:train_number]
    # train_datalist = all_paths[:-test_number]
    val_datalist = all_paths[train_number: train_number + val_number]
    test_datalist = all_paths[-test_number:]
    train_datalist = train_datalist + val_datalist

    print(f"training data is {len(train_datalist)}")
    print(f"validation data is {len(val_datalist)}")
    print(f"test data is {len(test_datalist)}", sorted(test_datalist))

    train_ds = BraTSDataset(train_datalist)
    val_ds = BraTSDataset(val_datalist)
    test_ds =  BraTSDataset(test_datalist)

    # loader = [train_ds, val_ds, test_ds]

    return train_ds, val_ds, test_ds



def get_single_process_loader_from_train(data_dir, patch_size, train_rate=0.7, val_rate=0.1, test_rate=0.2, seed=42):
    ## training all labeled data
    ## fold denote the validation data in training data
    all_paths = glob.glob(f"{data_dir}/*.npz")

    train_number = int(len(all_paths) * train_rate)
    val_number = int(len(all_paths) * val_rate)
    # test_number = int(len(all_paths) * test_rate)
    test_number = len(all_paths) - train_number - val_number
    random.seed(seed)
    random.shuffle(all_paths)

    train_datalist = all_paths[:train_number]
    val_datalist = all_paths[train_number: train_number + val_number]
    test_datalist = all_paths[-test_number:]

    print(f"training data is {len(train_datalist)}")
    print(f"validation data is {len(val_datalist)}")
    print(f"test data is {len(test_datalist)}", sorted(test_datalist))

    train_ds = SingleProcessDataset(train_datalist, patch_size)
    val_ds = SingleProcessDataset(val_datalist, patch_size)
    test_ds = SingleProcessDataset(test_datalist, patch_size)

    # loader = [train_ds, val_ds, test_ds]

    return train_ds, val_ds, test_ds

def get_unpacking_brats_loader_from_train(data_dir, train_rate=0.7, val_rate=0.1, test_rate=0.2, seed=42):
    ## training all labeled data
    ## fold denote the validation data in training data
    all_paths = glob.glob(f"{data_dir}/*.npy")
    # replace 保证shuffle的一致性
    all_paths = [path.replace(".npy", ".npz") for path in all_paths if not path.endswith('_seg.npy')]
    train_number = int(len(all_paths) * train_rate)
    val_number = int(len(all_paths) * val_rate)
    test_number = int(len(all_paths) * test_rate)
    random.seed(seed)
    random.shuffle(all_paths)

    train_datalist = all_paths[:train_number]
    val_datalist = all_paths[train_number: train_number + val_number]
    test_datalist = all_paths[-test_number:]

    print(f"training data is {len(train_datalist)}")
    print(f"validation data is {len(val_datalist)}")
    print(f"test data is {len(test_datalist)}", sorted(test_datalist))

    train_ds = BraTSDataset(train_datalist, need_unpacking=False)
    val_ds = BraTSDataset(val_datalist, need_unpacking=False)
    test_ds = BraTSDataset(test_datalist, need_unpacking=False)

    # loader = [train_ds, val_ds, test_ds]

    return train_ds, val_ds, test_ds
