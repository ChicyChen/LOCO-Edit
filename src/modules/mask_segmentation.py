from transformers import pipeline
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import gc, os
import torch
import torch.nn.functional as F

class SAM(object):
    def __init__(self, args, log_dir):
        self.generator = pipeline("mask-generation", model=args.mask_model_name, device= args.device, torch_dtype=torch.float32, cache_dir = args.cache_folder)
        self.args = args
        self.log_dir = os.path.join(log_dir, "mask")
        self.transparency = 0.4
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def mask_segmentation(self, image, resolution = 64):
        outputs = self.generator(image, points_per_batch=64)
        masks = outputs["masks"]
        # self.show_masks_on_image(image, masks)
        self.show_masks_on_image(image, masks)
        masks = torch.tensor(masks)
        masks = torch.round(F.interpolate(masks.unsqueeze(dim=1).to(torch.float32), [resolution, resolution]).squeeze(dim=1)).to(torch.bool)
        torch.save(masks, os.path.join(self.log_dir, "mask.pt"))
        return masks

    def color_mask(self, mask, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3)], axis=0) * 255
        else:
            color = np.array([30 , 144 , 255 ])
        h, w = mask.shape[-2:]
        color_mask = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        return color_mask

    def show_masks_on_image(self, raw_image, masks):
        image = np.array(raw_image)
        image_totalmask = image.copy()
        for idx, mask in enumerate(masks):
            if mask.sum() > self.args.filter_mask:
                color_mask = self.color_mask(mask, random_color=True)
                image_per_mask = image.copy()
                image_per_mask[mask] = ((image_per_mask[mask].astype(np.float32) * (1 - self.transparency) + color_mask[mask].astype(np.float32) * self.transparency)/2).astype(np.uint8)
                image_totalmask[mask] = ((image_totalmask[mask].astype(np.float32) * (1 - self.transparency) + color_mask[mask].astype(np.float32) * self.transparency)/2).astype(np.uint8)
                plt.imsave(os.path.join(self.log_dir, f"mask_{idx}.png"), image_per_mask)
        plt.imsave(os.path.join(self.log_dir, "total_mask.png"), image_totalmask)
