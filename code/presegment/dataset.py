import hashlib

import numpy as np
import torchvision
from matplotlib import pyplot as plt
from pandas import DataFrame
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
from PIL import Image
from torchvision.transforms import transforms

from common.path_util import data_dir
from common.plot_util import save_next
import lightning.pytorch as pl


class TeethDataset(Dataset):
    """ Segmentation dataset """

    def __init__(
            self,
            paths: DataFrame,
            size=(384, 384),
            blur_kernel_size=51,
            blur_sigma=0.5,
            normalize_top_bottom=0.03,
    ):
        self.size = size
        self.img_resizer = torchvision.transforms.Resize(size=self.size)
        self.mask_resizer = torchvision.transforms.Resize(
            size=self.size,
            interpolation=torchvision.transforms.InterpolationMode.NEAREST
        )
        self.blur = transforms.GaussianBlur(blur_kernel_size, blur_sigma)

        self.normalize_top_bottom = normalize_top_bottom

        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def normalize(self, img):
        """ Normalize image.
        This includes removing the top and bottom x% of intensities for burn-in and burn-out correction.
        """
        intensities = torch.flatten(img)
        intensities = torch.sort(intensities).values
        idx = int(intensities.size(0) * self.normalize_top_bottom)
        newmin = intensities[idx]
        newmax = intensities[-idx]
        img = (img - newmin) / (newmax - newmin)
        img[img < 0] = 0
        img[img > 1] = 1
        return img

    def __getitem__(self, item):
        row = self.paths.iloc[item]
        raw_img = self.normalize(torch.tensor(np.array(Image.open(row['image']))))
        if raw_img.shape[-1] == 3:
            raw_img = raw_img[..., 0]

        mask_path = row['mask']
        if mask_path is None:
            mask = torch.zeros(raw_img.shape, dtype=torch.bool)
        else:
            mask = torch.tensor(np.array(Image.open(mask_path)).astype(bool))
            if mask.shape[-1] == 3:
                mask = mask[..., 0]

        raw_img = self.img_resizer(raw_img[None])
        img = self.blur(raw_img[None])[0]
        mask = self.img_resizer(mask[None]) # TODO why does this work

        id_ = row['id']

        return raw_img, img, mask, id_


class TeethDataModule(pl.LightningDataModule):

    @staticmethod
    def filter_duplicates(images, masks):
        # Dictionary to store file hashes
        hashes = {}
        duplicates = []

        result_images = []
        result_masks = []

        for i, (image_path, mask_path) in enumerate(zip(images, masks)):
            with open(image_path, 'rb') as f:
                image_hash = hashlib.sha256(f.read()).hexdigest()
            if image_hash in hashes:
                duplicates.append((str(image_path), str(hashes[image_hash])))
            else:
                hashes[image_hash] = image_path
                result_images.append(image_path)
                result_masks.append(mask_path)

        return result_images, result_masks, duplicates

    def __init__(self, batch_size, num_workers):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        images = []
        masks = []

        # Add labelled data

        ## Adult tooth segmentation / train

        current_images_root = data_dir / "Adult tooth segmentation dataset" / "Dataset and code" / "train" / "images"
        current_masks_root = data_dir / "Adult tooth segmentation dataset" / "Dataset and code" / "train" / "masks"

        current_images = sorted(list(current_images_root.iterdir()))
        current_masks = [current_masks_root / (img.stem + '.bmp') for img in current_images]
        images.extend(current_images)
        masks.extend(current_masks)

        # TODO include other sets as well

        # Deduplicate

        images, masks, duplicates = self.filter_duplicates(images, masks)
        print('Duplicates:')
        for d in duplicates:
            print(d)
        print('End of duplicates')

        self.paths = pd.DataFrame({
            'image': images,
            'mask': masks,
        })
        print(self.paths)

        # Split cases

        indices = np.arange(len(self.paths))

        np.random.seed(42)
        np.random.shuffle(indices)
        index_1 = int(len(indices) * 0.8)
        index_2 = int(len(indices) * 0.9)
        train_cases = indices[:index_1]
        val_cases = indices[index_1:index_2]
        test_cases = indices[index_2:]

        # Add unlabelled data
        current_images = []

        ## Add Panoramic Dental Dataset (no mask)
        current_images_root = data_dir / "Panoramic Dental Dataset" / "images"
        current_images.extend(current_images_root.iterdir())

        ## Add random test images (no mask)
        current_images_root = data_dir / "Other"
        current_images.extend(current_images_root.iterdir())

        ## Add Roboflow images (no mask)
        current_images_root = data_dir / "Roboflow"
        current_images.extend(current_images_root.glob('**/*.jpg'))

        current_images.sort()
        current_masks = [None for _ in current_images]
        _, _, duplicates = self.filter_duplicates(images + current_images, masks + current_masks)
        print(f'Duplicates found in prediction set:', len(duplicates))
        predict_df = pd.DataFrame({
            'image': current_images,
            'mask': current_masks,
        })
        self.paths = pd.concat([self.paths, predict_df], ignore_index=True)
        pred_cases = np.arange(len(self.paths) - len(predict_df), len(self.paths))

        # Finally
        self.paths['id'] = range(len(self.paths))

        self.train_data = TeethDataset(self.paths.iloc[train_cases])
        self.val_data = TeethDataset(self.paths.iloc[val_cases])
        self.test_data = TeethDataset(self.paths.iloc[test_cases])
        self.predict_data = TeethDataset(self.paths.iloc[pred_cases])
        print('Train length:', len(self.train_data))
        print('Val length:', len(self.val_data))
        print('Test length:', len(self.test_data))
        print('Predict length:', len(self.predict_data))

    def __len__(self):
        return len(self.paths)

    def train_dataloader(self, batch_size: int = None):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size if batch_size is None else batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self, batch_size: int = None):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size if batch_size is None else batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self, batch_size: int = None):
        return DataLoader(
            self.test_data,
            batch_size=1 if batch_size is None else batch_size,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self, batch_size: int = None):
        return DataLoader(
            self.predict_data,
            batch_size=1 if batch_size is None else batch_size,
            num_workers=self.num_workers,
        )


def main():
    """ Test the dataset"""
    dm = TeethDataModule(1, 1)
    dm.setup('fit')

    print('Dataset length:', len(dm))

    for raw_img, img, mask, id_ in dm.test_dataloader():
        fig, ax = plt.subplots()  # type: plt.Figure, plt.Axes
        print(raw_img.shape, img.shape, mask.shape)
        img = torch.stack([img[0, 0]] * 3, dim=-1) * 0.7
        img[..., 0] += mask[0, 0] * 0.3
        ax.imshow(img)
        save_next(fig, 'test_segment', with_fig_num=False)
        break


if __name__ == '__main__':
    main()
