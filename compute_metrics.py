from monai.utils import set_determinism
import torch
import os
import numpy as np
import SimpleITK as sitk
from medpy import metric
import argparse
from tqdm import tqdm

import numpy as np

from datasets.brats import get_brats_loader_from_train

set_determinism(123)

# parser = argparse.ArgumentParser()
#
# parser.add_argument("--pred_name", required=True, type=str)

# args = parser.parse_args()
#
# pred_name = args.pred_name


def cal_metric(gt, pred, voxel_spacing):
    dice = metric.binary.dc(pred, gt)
    if gt.sum() > 0 and dice > 0:
        hd95 = metric.binary.hd95(pred, gt, voxelspacing=voxel_spacing)
        return np.array([dice, hd95])
    elif pred.sum() == 0 and gt.sum() == 0:
        return np.array([1.0, 0])
    else:
        return np.array([0, 50])


def each_cases_metric(gt, pred, voxel_spacing, classes_num = 3):
    class_wise_metric = np.zeros((classes_num, 2))
    if len(gt.shape) == 4:
        for cls in range(0, classes_num):
            class_wise_metric[cls, ...] = cal_metric(pred[cls], gt[cls], voxel_spacing)
    else:
        class_wise_metric[0, ...] = cal_metric(pred, gt, voxel_spacing)
    # print(class_wise_metric)
    return class_wise_metric

def each_cases_metric_aiib(gt, pred, voxel_spacing, classes_num = 1):
    class_wise_metric = np.zeros((classes_num, 1))
    class_wise_metric[0, ...] = metric.binary.jc(pred, gt)
    print(class_wise_metric)
    return class_wise_metric


def convert_labels(labels):
    ## TC, WT and ET
    labels = labels.unsqueeze(dim=0)

    result = [(labels == 1) | (labels == 3), (labels == 1) | (labels == 3) | (labels == 2), labels == 3]

    return torch.cat(result, dim=0).float()


def convert_prostate_labels(labels, reverse=False):
    labels = labels.unsqueeze(dim=0)
    if reverse:
        result = [labels == 2, labels == 1]
    else:
        result = [labels == 1, labels == 2]

    return torch.cat(result, dim=0).float()


if __name__ == "__main__":
    DATA_TYPE = "prostate"
    results_root = "prediction_results"

    if "brats" in DATA_TYPE.lower():
        data_dir = "/home/caoyitong/DataProjects/brats2023/data/fullres/train"
        raw_data_dir = "/home/caoyitong/DataProjects/BraTS2023-GLI/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
        pred_name = "segrwkv-brats"
        classes_num = 3
        metric_num = 2
        seed = 42
    elif "aiib" in DATA_TYPE.lower():
        data_dir = "/home/caoyitong/DataProjects/AIIB2023/data/fullres/train"
        raw_data_dir = "/home/caoyitong/DataProjects/AIIB2023/AIIB23_Train_T1/gt"
        pred_name = "segrwkv-aiib"
        classes_num = 1
        metric_num = 1
        seed = 42
    elif "prostate" in DATA_TYPE.lower():
        data_dir = "/home/caoyitong/DataProjects/prostate/data/fullres/train"
        raw_data_dir = "/home/caoyitong/DataProjects/prostate/labelsTr"
        pred_name = "segrwkv-prostate"
        classes_num = 2
        metric_num = 2
        seed = 7

    train_ds, val_ds, test_ds = get_brats_loader_from_train(data_dir, seed=seed)
    data_num = len(test_ds)
    all_results = np.zeros((data_num, classes_num, metric_num))

    ind = 0
    for batch in tqdm(test_ds, total=len(test_ds)):
        properties = batch["properties"]
        case_name = properties["name"]
        if "brats" in DATA_TYPE.lower():
            gt_itk = os.path.join(raw_data_dir, case_name, f"seg.nii.gz")
        elif "aiib" in DATA_TYPE.lower() or "prostate" in DATA_TYPE.lower():
            gt_itk = os.path.join(raw_data_dir, case_name + f".nii.gz")
        voxel_spacing = [1, 1, 1]
        gt_itk = sitk.ReadImage(gt_itk)
        gt_array = sitk.GetArrayFromImage(gt_itk).astype(np.int32)
        gt_array = torch.from_numpy(gt_array)
        if "brats" in DATA_TYPE.lower():
            gt_array = convert_labels(gt_array).numpy()
        elif "aiib" in DATA_TYPE:
            gt_array = gt_array.numpy()
        elif "prostate" in DATA_TYPE.lower():
            gt_array = convert_prostate_labels(gt_array).numpy()

        pred_itk = sitk.ReadImage(f"./{results_root}/{pred_name}/{case_name}.nii.gz")
        pred_array = sitk.GetArrayFromImage(pred_itk)
        if "aiib" in DATA_TYPE.lower():
            m = each_cases_metric_aiib(gt_array, pred_array, voxel_spacing)
        else:
            m = each_cases_metric(gt_array, pred_array, voxel_spacing, classes_num)

        all_results[ind, ...] = m

        ind += 1

    os.makedirs(f"./{results_root}/result_metrics/", exist_ok=True)
    np.save(f"./{results_root}/result_metrics/{pred_name}.npy", all_results)

    result = np.load(f"./{results_root}/result_metrics/{pred_name}.npy")
    print(result.shape)
    print(result.mean(axis=0))
    print(result.std(axis=0))