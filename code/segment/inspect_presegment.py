import numpy as np
import torch
from dataset import CariesDataModule

dm = CariesDataModule(1, 1)
dm.setup('fit')

unet_ratios = []
segformer_ratios = []
total_ratios = []
for raw_img, fany_unet_pred, segformer_pred, img, mask in dm.test_dataloader():
    unet_ratio = (fany_unet_pred * mask).sum() / mask.sum()
    segformer_ratio = (segformer_pred * mask).sum() / mask.sum()
    total_ratio = (torch.maximum(segformer_pred, fany_unet_pred) * mask).sum() / mask.sum()
    unet_ratios.append(unet_ratio.detach().cpu().numpy())
    segformer_ratios.append(segformer_ratio.detach().cpu().numpy())
    total_ratios.append(total_ratio.detach().cpu().numpy())

unet_coverage = np.array(unet_ratios).mean()
segformer_coverage = np.array(segformer_ratios).mean()
total_coverage = np.array(total_ratios).mean()

print(f"Unet caries coverage: {unet_coverage}\nSegformer caries coverage: {segformer_coverage}\nTotal caries coverage: {total_coverage}\n")