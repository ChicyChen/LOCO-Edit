import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchmetrics.regression import MeanSquaredError as MSE
from glob import glob as glob
import os
import argparse
import numpy as np

from PIL import Image 
import torchvision.transforms as transforms

def parse_args():
    parser = argparse.ArgumentParser()

    # default setting 
    parser.add_argument('--eval_metric', type=str, default='ssim', required=False)

    args = parser.parse_args()
    return args

# FID for similarity between two distributions: image quality for overall editing quality

# SSIM for similarity between pairs of images
# A value closer to 1 indicates better image quality
def eval_SSIM(preds, target):
    ssim = SSIM()
    result = ssim(preds, target)
    return result

# LPIPS for similarity between pairs of images
# Lower means more similar
def eval_LPIPS(preds, target):
    lpips = LPIPS()
    result = ssim(preds, target)
    return result

# masked MSE while editting using masks for similarity between pairs of masked images
def eval_MMSE(preds, target, mask):
    mmse = MSE()
    result = ssim(preds[mask], target[mask])
    return result

def main():
    args = parse_args()
    if args.eval_metric == "ssim":
        metric_op = SSIM()
    elif args.eval_metric == "lpips":
        metric_op = LPIPS()
    elif args.eval_metric == "mmse":
        metric_op = MSE()
    else:
        raise Exception("not implemented")

    folder_preds = ""
    folder_original = ""
    preds_path = glob(os.path.join(folder_preds, '*.png'))
    preds_path.sort()
    target_path = glob(os.path.join(folder_original, '*.png'))
    target_path.sort()

    result_list = []
    for idx in range(len(preds_path)):
        preds = preds_path[idx]
        target = target_path[idx]
        
        if os.path.basename(preds) != os.path.basename(target):
            raise Exception("pairs not match")

        img_idx = int(basename(preds)[:-4])
        mask_path = os.path.join(folder_preds, "mask", img_idx)

        preds = np.array(Image.open(preds)).astype(np.float32)
        target = np.array(Image.open(target)).astype(np.float32)
        preds = torch.from_numpy(preds).permute(2,0,1).unsqueeze(0)
        target = torch.from_numpy(target).permute(2,0,1).unsqueeze(0)
        if args.eval_metric == "mmse":
            mask = load(mask_path)
            result = metric_op(preds,targetm,mask)
        result = metric_op(preds,target)
        result_list.append(result)

    ave_result = sum(result_list) / len(result_list) 
    print(args.eval_metric, "result:", ave_result)


if __name__ == "__main__":
    main()
