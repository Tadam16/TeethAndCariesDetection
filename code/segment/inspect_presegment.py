import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from common.plot_util import save_next
from dataset import CariesDataModule

dm = CariesDataModule(1, 1)
dm.setup('fit')

unet_ratios = []
segformer_ratios = []
total_ratios = []
num_bad_predictions = 0
for raw_img, fancy_unet_pred, segformer_pred, img, mask in tqdm(dm.test_dataloader()):
    fancy_unet_pred= fancy_unet_pred >= 0.5
    segformer_pred = segformer_pred >= 0.5
    unet_ratio = (fancy_unet_pred * mask).sum() / mask.sum()
    segformer_ratio = (segformer_pred * mask).sum() / mask.sum()
    total_ratio = (torch.maximum(segformer_pred, fancy_unet_pred) * mask).sum() / mask.sum()
    unet_ratios.append(unet_ratio.detach().cpu().numpy())
    segformer_ratios.append(segformer_ratio.detach().cpu().numpy())
    total_ratios.append(total_ratio.detach().cpu().numpy())

    if total_ratio < 0.3:
        num_bad_predictions += 1
        if num_bad_predictions < 20:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 10))

            img1 = np.stack([img.detach().cpu().numpy()[0, 0]] * 3, axis=-1) * 0.7
            img1[..., 1] += fancy_unet_pred.detach().cpu().numpy()[0, 0] * 0.3
            img1[..., 2] += segformer_pred.detach().cpu().numpy()[0, 0] * 0.3
            ax1.imshow(img1)
            ax1.set_title(f'Input')

            img2 = np.stack([img.detach().cpu().numpy()[0, 0]] * 3, axis=-1) * 0.7
            img2[..., 0] += mask.detach().cpu().numpy()[0, 0] * 0.3
            ax2.imshow(img2)
            ax2.set_title(f'Ground truth: unet_ratio={unet_ratio:.2f}, '
                          f'segformer_ratio={segformer_ratio:.2f}, '
                          f'total_ratio={total_ratio:.2f}')

            save_next(fig, 'bad_prediction')

unet_coverage = np.array(unet_ratios).mean()
segformer_coverage = np.array(segformer_ratios).mean()
total_coverage = np.array(total_ratios).mean()

print(f"Unet caries coverage: {unet_coverage}")
print(f"Segformer caries coverage: {segformer_coverage}")
print(f"Total caries coverage: {total_coverage}")
print(f"Bad predictions: {num_bad_predictions}")
print(f"Bad prediction ratio: {num_bad_predictions / len(dm.test_dataloader())}")