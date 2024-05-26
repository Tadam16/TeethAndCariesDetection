import hashlib
import json

import numpy as np
import torchvision
import yaml
from matplotlib import pyplot as plt
from pandas import DataFrame
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
from PIL import Image
from torchvision.transforms import transforms

from common.path_util import data_dir, presegment_out_dir, preprocess_out_dir, rebase
from common.plot_util import save_next
import lightning.pytorch as pl

from roboflow_preprocess.filter import BoundingBox


class CariesDataset(Dataset):
    """ Segmentation dataset """

    def __init__(
            self,
            paths: DataFrame,
            size,
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

        fancy_unet_pred = torch.tensor(np.load(row['fancy_unet_pred']))
        segformer_pred = torch.tensor(np.load(row['segformer_pred']))

        if row['bbox'] is not None:
            bbox = BoundingBox.json_to_list(row['bbox'])

            rows, columns = raw_img.shape
            mask = mask[bbox[0]:bbox[1], bbox[2]:bbox[3]]
            raw_img = raw_img[bbox[0]:bbox[1], bbox[2]:bbox[3]]

            pred_rows, pred_columns = fancy_unet_pred.shape
            fancy_unet_box = int(bbox[0] * pred_rows / rows), int(bbox[1] * pred_rows / rows), int(
                bbox[2] * pred_columns / columns), int(bbox[3] * pred_columns / columns)
            fancy_unet_pred = fancy_unet_pred[fancy_unet_box[0]:fancy_unet_box[1], fancy_unet_box[2]:fancy_unet_box[3]]

            pred_rows, pred_columns = segformer_pred.shape
            segformer_box = int(bbox[0] * pred_rows / rows), int(bbox[1] * pred_rows / rows), int(
                bbox[2] * pred_columns / columns), int(bbox[3] * pred_columns / columns)
            segformer_pred = segformer_pred[segformer_box[0]:segformer_box[1], segformer_box[2]:segformer_box[3]]

        raw_img = self.img_resizer(raw_img[None])
        fancy_unet_pred = self.img_resizer(fancy_unet_pred[None])
        segformer_pred = self.img_resizer(segformer_pred[None])
        img = self.blur(raw_img[None])[0]
        mask = self.img_resizer(mask[None])

        return raw_img, fancy_unet_pred, segformer_pred, img, mask


class CariesDataModule(pl.LightningDataModule):

    @staticmethod
    def filter_duplicates(images, *other_lists):
        # Dictionary to store file hashes
        hashes = {}
        duplicates = []

        result_images = []
        result_other_data = []

        for i, (image_path, *other_data) in enumerate(zip(images, *other_lists)):
            with open(image_path, 'rb') as f:
                image_hash = hashlib.sha256(f.read()).hexdigest()
            if image_hash in hashes:
                duplicates.append((str(image_path), str(hashes[image_hash])))
            else:
                hashes[image_hash] = image_path
                result_images.append(image_path)
                result_other_data.append(other_data)

        return (
            result_images,
            duplicates,
            *(list(it) for it in zip(*result_other_data))
        )

    def __init__(self, batch_size, num_workers, image_size=(384,384)):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size

        images = []
        masks = []
        bboxes = []

        # Add labelled data

        ## Panoramic Dental Dataset

        current_images_root = data_dir / "Panoramic Dental Dataset" / "images"
        current_masks_root = data_dir / "Panoramic Dental Dataset" / "labels"

        current_images = sorted(list(current_images_root.iterdir()))
        current_masks = [current_masks_root / img.name for img in current_images]
        images.extend(current_images)
        masks.extend(current_masks)
        bboxes.extend([None] * len(current_images))

        ## Roboflow datasets

        current_images_root = data_dir / 'Roboflow'
        current_masks_root = preprocess_out_dir / 'Roboflow'

        def get_current_masks_and_images():
            # with open('/out/preprocess/filter/train.yaml') as f:
            #     non_augmented_train = {
            #         it['path']: it['bbox']
            #         for it
            #         in yaml.load(f, Loader=yaml.FullLoader)
            #     }
            with open('/out/preprocess/filter/valid.yaml') as f:
                valid = {
                    it['path']: it['bbox']
                    for it
                    in yaml.load(f, Loader=yaml.FullLoader)
                }
            with open('/out/preprocess/filter/test.yaml') as f:
                test = {
                    it['path']: it['bbox']
                    for it
                    in yaml.load(f, Loader=yaml.FullLoader)
                }

            masks = sorted(list(current_masks_root.glob('**/*.png')))
            images = [rebase(current_masks_root, current_images_root, mask).parent / f'{mask.stem}.jpg'
                      for mask in masks]

            result_masks, result_images, result_bboxes = [], [], []
            for mask, image in zip(masks, images):
                bbox = None
                if not image.exists():
                    print(f'Image {image} does not exist')
                    continue
                if not mask.exists():
                    print(f'Mask {mask} does not exist')
                    continue
                if image.parent.name == 'train':
                    # if str(image) in non_augmented_train:
                    #     bbox = non_augmented_train[str(image)]
                    #     assert bbox is not None
                    # else:
                    continue
                if image.parent.name == 'valid':
                    if str(image) in valid:
                        bbox = valid[str(image)]
                if image.parent.name == 'test':
                    if str(image) in test:
                        bbox = test[str(image)]
                result_images.append(image)
                result_masks.append(mask)
                result_bboxes.append(bbox)

            return result_images, result_masks, result_bboxes

        current_images, current_masks, current_bboxes = get_current_masks_and_images()
        images.extend(current_images)
        masks.extend(current_masks)
        bboxes.extend(current_bboxes)

        # Deduplicate

        images, duplicates, masks, bboxes = self.filter_duplicates(images, masks, bboxes)
        print('Duplicates:')
        for d in duplicates:
            print(d)
        print('End of duplicates')

        self.paths = pd.DataFrame({
            'image': images,
            'mask': masks,
            'bbox': bboxes,
        })

        # Add presegmentations to the dataset

        def locate_pred(image, model_name):
            old_base = data_dir
            new_base = presegment_out_dir / 'predictions' / model_name / 'predict'
            relative_path = image.relative_to(old_base)
            new_path = (new_base / relative_path).resolve()
            return new_path.parent / f'{new_path.stem}.npy'

        self.paths['fancy_unet_pred'] = self.paths['image'].apply(lambda x: locate_pred(x, 'fancy_unet'))
        self.paths['segformer_pred'] = self.paths['image'].apply(lambda x: locate_pred(x, 'segformer'))
        self.paths['id'] = range(len(self.paths))
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

        # TODO Add unlabelled data

        # Finally

        self.train_data = CariesDataset(self.paths.iloc[train_cases], size=image_size)
        self.val_data = CariesDataset(self.paths.iloc[val_cases], size=image_size)
        self.test_data = CariesDataset(self.paths.iloc[test_cases], size=image_size)
        print('Train length:', len(self.train_data))
        print('Val length:', len(self.val_data))
        print('Test length:', len(self.test_data))

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


def main():
    """ Test the dataset"""
    dm = CariesDataModule(1, 1)
    dm.setup('fit')

    print('Dataset length:', len(dm))

    for i, (raw_img, fancy_unet_pred, segformer_pred, img, mask) in enumerate(dm.test_dataloader()):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))

        img1 = np.stack([raw_img.detach().cpu().numpy()[0, 0]] * 3, axis=-1)
        ax1.imshow(img1)
        ax2.set_title(f'Raw')

        img2 = np.stack([img.detach().cpu().numpy()[0, 0]] * 3, axis=-1) * 0.7
        img2[..., 1] += fancy_unet_pred.detach().cpu().numpy()[0, 0] * 0.3
        img2[..., 2] += segformer_pred.detach().cpu().numpy()[0, 0] * 0.3
        ax2.imshow(img2)
        ax2.set_title(f'Input')

        img3 = np.stack([img.detach().cpu().numpy()[0, 0]] * 3, axis=-1) * 0.7
        img3[..., 0] += mask.detach().cpu().numpy()[0, 0] * 0.3
        ax3.imshow(img3)
        ax3.set_title(f'Ground truth')

        save_next(fig, 'test_segment')
        plt.close(fig)

        if i == 10:
            break


if __name__ == '__main__':
    main()
