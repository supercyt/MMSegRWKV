import glob
import os

import numpy as np
import SimpleITK as sitk
import torch

from tqdm import tqdm
import time
from torch.multiprocessing import Pool

from compute_metrics import convert_labels, convert_prostate_labels, each_cases_metric, each_cases_metric_aiib
from datasets.brats import get_brats_loader_from_train
from predict import BraTSPrector

DATA_TYPE = "inhouse"
results_root = "prediction_results"

augmentation = True

max_epoch = 600
batch_size = 2
val_every = 10
num_gpus = 1
device = "cuda:0"
architecture = "mmsegrwkv"


if "brats" in DATA_TYPE.lower():
    data_dir = f""
    model_dir = f"./logs/segrwkv-brats2023/{architecture}/"
    save_path = f"./{results_root}/segrwkv-brats/{architecture}"
    in_chans = 4
    out_chans = 4
    seed = 42
    raw_data_dir = ""
    pred_name = "segrwkv-brats"
    classes_num = 3
    metric_num = 2
    roi_size = [128, 128, 128]
elif "isles" in DATA_TYPE.lower():
    data_dir = f""
    model_dir = f"./logs/segrwkv-isles2022/{architecture}/"
    save_path = f"./{results_root}/segrwkv-isles/{architecture}"
    in_chans = 3
    out_chans = 2
    seed = 130
    raw_data_dir = ""
    pred_name = "segrwkv-isles"
    classes_num = 1
    metric_num = 2
    roi_size = [64, 64, 64]
elif "inhouse" in DATA_TYPE.lower():
    data_dir = f""
    model_dir = f"./logs/segrwkv-inhouse/{architecture}/"
    save_path = f"./{results_root}/segrwkv-inhouse/{architecture}"
    in_chans = 2
    out_chans = 2
    seed = 42
    raw_data_dir = ""
    pred_name = "segrwkv-inhouse"
    classes_num = 1
    metric_num = 2
    roi_size = [16, 128, 128]
elif "prostate" in DATA_TYPE.lower():
    data_dir = f""
    model_dir = f"./logs/segrwkv-prostate/{architecture}/"
    save_path = f"./prediction_results/segrwkv-prostate/{architecture}"
    in_chans = 2
    out_chans = 3
    seed = 7330479
    raw_data_dir = ""
    pred_name = "segrwkv-prostate"
    classes_num = 2
    metric_num = 2
    roi_size = [32, 128, 128]


def single_process_eval(model_path):
    train_ds, val_ds, test_ds = get_brats_loader_from_train(data_dir, seed=seed)
    print("model name: ", model_path)
    metric_save_name = pred_name + "-" + model_path.split("/")[-1].rstrip(".pt")
    pred_save_path = save_path + "/" + model_path.split("/")[-1].rstrip(".pt")
    validator = BraTSPrector(
        architecture=architecture,
        max_epochs=max_epoch,
        batch_size=batch_size,
        in_chans=in_chans,
        out_chans=out_chans,
        model_path=model_path,
        save_path=pred_save_path,
        device=device,
        logdir="",
        val_every=val_every)

    validator.validation_single_gpu(test_ds)

    data_num = len(test_ds)
    all_results = np.zeros((data_num, classes_num, metric_num))

    ind = 0
    for batch in test_ds:
        properties = batch["properties"]
        case_name = properties["name"]
        if "brats" in DATA_TYPE.lower():
            gt_itk = os.path.join(raw_data_dir, case_name, f"seg.nii.gz")
        elif "prostate" in DATA_TYPE.lower():
            gt_itk = os.path.join(raw_data_dir, case_name + f".nii.gz")
        elif "inhouse" in DATA_TYPE.lower():
            gt_itk = os.path.join(raw_data_dir, case_name + f".nii")
        elif "isles" in DATA_TYPE.lower():
            gt_itk = os.path.join(raw_data_dir, case_name + f"/ses-0001/{case_name}_ses-0001_msk.nii.gz")
        voxel_spacing = [1, 1, 1]
        gt_itk = sitk.ReadImage(gt_itk)
        gt_array = sitk.GetArrayFromImage(gt_itk).astype(np.int32)
        gt_array = torch.from_numpy(gt_array)
        if "brats" in DATA_TYPE.lower():
            gt_array = convert_labels(gt_array).numpy()
        elif "isles" in DATA_TYPE or "inhouse" in DATA_TYPE:
            gt_array = gt_array.numpy()
        elif "prostate" in DATA_TYPE.lower():
            gt_array = convert_prostate_labels(gt_array).numpy()

        pred_itk = sitk.ReadImage(f"./{pred_save_path}/{case_name}.nii.gz")
        pred_array = sitk.GetArrayFromImage(pred_itk)

        m = each_cases_metric(gt_array, pred_array, voxel_spacing, classes_num)
        # if m[0, 0] < 0.5:
        print(case_name, m[0, 0])
        all_results[ind, ...] = m

        ind += 1

    os.makedirs(f"./{results_root}/result_metrics/{DATA_TYPE}/{architecture}", exist_ok=True)
    np.save(f"./{results_root}/result_metrics/{DATA_TYPE}/{architecture}/{metric_save_name}.npy", all_results)

    result = np.load(f"./{results_root}/result_metrics/{DATA_TYPE}/{architecture}/{metric_save_name}.npy")
    print(f"{metric_save_name} dice:", result.mean(axis=0))
    # print(f"{pred_name}", result.std(axis=0))
    # del validator


if __name__ == "__main__":
    model_paths = glob.glob(model_dir + "*.pt")
    model_paths = list(filter(lambda x: float(x.split("_")[-1].rstrip(".pt")) > 0.73, model_paths))
    for model_path in model_paths:
        single_process_eval(model_path)
    # t_start = time.time()
    # torch.multiprocessing.set_start_method('spawn')
    # pool = Pool(4)
    
    # for model_path in model_paths:
    #     pool.apply_async(func=single_process_eval, args=(model_path,))  # 维持执行的进程总数为processes，当一个进程执行完毕后会添加新的进程进去
    
    # pool.close()
    # pool.join()  # 进程池中进程执行完毕后再关闭，如果注释，那么程序直接关闭。
    # pool.terminate()
    # t_end = time.time()
    # t = t_end - t_start
    # print('the program time is :%s' % t)