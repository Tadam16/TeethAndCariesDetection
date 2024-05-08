import argparse
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.io import savemat

import common.segformer as segformer
from dataset import CariesDataModule
import torch.utils.data
from common.fancy_unet import Unet as FancyUnet
from common.path_util import segment_out_dir

import torchmetrics
import lightning.pytorch as pl
from scipy.spatial.distance import directed_hausdorff


class BaseModel(pl.LightningModule):
    """ Common settings for the two models """

    def __init__(self, internal):
        super().__init__()

        self.internal = internal

        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(30.0))
        self.dice_score_fn = self.soft_dice_score_fn

        self.dice_frequency = 32  # MUST be min accumulate_grad_batches and SHOULD be equal

    def soft_dice_score_fn(self, pred, mask, eps=1e-7):
        """ Calculate soft dice score """
        pred = pred.view(-1)
        mask = mask.view(-1)
        intersection = (pred * mask).sum()
        return (2. * intersection) / (pred.sum() + mask.sum() + eps)

    @staticmethod
    def hausdorff(preds, gts):
        dsts = []
        num_inf = 0
        for i in range(preds.shape[0]):
            pred = preds[i,0]
            gt = gts[i,0]
            pred_coords = np.stack(np.where(pred > 0.5), -1)
            gt_coords = np.stack(np.where(gt), -1)
            dist = max(directed_hausdorff(pred_coords, gt_coords)[0], directed_hausdorff(gt_coords, pred_coords)[0])
            if np.isinf(dist):
                num_inf += 1
            else:
                dsts.append(dist)
        return torch.tensor(np.array(dsts)), num_inf

    def forward(self, x):
        return self.internal(x)

    def predict_step(self, batch, batch_idx):
        raw_img, fancy_unet_pred, segformer_pred, img, mask = batch
        input_ = torch.cat((img, fancy_unet_pred, segformer_pred), dim=1)
        pred_raw = self(input_)
        pred = torch.sigmoid(pred_raw)
        return raw_img, pred, fancy_unet_pred, segformer_pred, img, mask

    def training_step(self, batch, batch_idx):
        """ Train, occasionally calculate metrics, occasionally show figures"""
        raw_img, fancy_unet_pred, segformer_pred, img, mask = batch
        input_ = torch.cat((img, fancy_unet_pred, segformer_pred), dim=1)
        pred_raw = self(input_)
        loss = self.loss_fn(pred_raw, mask.float())
        self.log(f'train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        if batch_idx % self.dice_frequency == 0:
            pred = torch.sigmoid(pred_raw)
            dice_score = self.dice_score_fn(pred, mask)
            self.log(f'train/dice_score', dice_score, on_step=True, on_epoch=True, prog_bar=True)

            if batch_idx % (self.dice_frequency * 10) == 0:
                self.show_example('train', raw_img, fancy_unet_pred, segformer_pred, mask, pred, batch_idx)

        return loss

    def validation_step(self, batch, batch_idx):
        """ Calculate metrics, show figures for 10 fixed images"""
        raw_img, fancy_unet_pred, segformer_pred, img, mask = batch
        input_ = torch.cat((img, fancy_unet_pred, segformer_pred), dim=1)
        pred_raw = self(input_)
        loss = self.loss_fn(pred_raw, mask.float())
        self.log(f'val/loss', loss, on_epoch=True)

        pred = torch.sigmoid(pred_raw)
        dice_score = self.dice_score_fn(pred, mask)
        self.log(f'val/dice_score', dice_score, on_epoch=True)

        if batch_idx <= 10:
            self.show_example('val', raw_img, fancy_unet_pred, segformer_pred, mask, pred, batch_idx)

    def test_step(self, batch, batch_idx):
        """ Calculate metrics including custom metric, show figures for 10 fixed images"""
        raw_img, fancy_unet_pred, segformer_pred, img, mask = batch
        input_ = torch.cat((img, fancy_unet_pred, segformer_pred), dim=1)
        pred_raw = self(input_)
        loss = self.loss_fn(pred_raw, mask.float())
        self.log(f'test/loss', loss, on_epoch=True)

        pred = torch.sigmoid(pred_raw)
        dice_score = self.dice_score_fn(pred, mask)
        hausdorff_distance, num_inf = self.hausdorff(pred.detach().cpu().numpy(), mask.detach().cpu().numpy())
        self.log(f'test/dice_score', dice_score, on_epoch=True)
        if hausdorff_distance.shape[0] > 0:
            self.log(f'test/hausdorff_distance', hausdorff_distance, on_epoch=True)
        self.log(f'test/hausdorff_distance_inf', num_inf, on_epoch=True, reduce_fx='sum')

        if batch_idx <= 10:
            self.show_example('test', raw_img, fancy_unet_pred, segformer_pred, mask, pred, batch_idx)

        return loss

    @staticmethod
    def get_fig(img, fancy_unet_pred, segformer_pred, mask, pred):
        """ Create and save figure with 3 images: input, prediction and ground truth """
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))

        img1 = np.stack([img.detach().cpu().numpy()[0, 0]] * 3, axis=-1) * 0.7
        img1[..., 1] += fancy_unet_pred.detach().cpu().numpy()[0, 0] * 0.3
        img1[..., 2] += segformer_pred.detach().cpu().numpy()[0, 0] * 0.3
        ax1.imshow(img1)
        ax1.set_title(f'Input')

        img2 = np.stack([img.detach().cpu().numpy()[0, 0]] * 3, axis=-1) * 0.7
        pred = pred.detach().cpu().numpy()
        img2[..., 0] += pred[0, 0] * 0.3
        ax2.imshow(img2)
        ax2.set_title(f'Prediction')

        img3 = np.stack([img.detach().cpu().numpy()[0, 0]] * 3, axis=-1) * 0.7
        img3[..., 0] += mask.detach().cpu().numpy()[0, 0] * 0.3
        ax3.imshow(img3)
        ax3.set_title(f'Ground truth')
        return fig

    def show_example(self, phase, raw_img, fancy_unet_pred, segformer_pred, mask, pred, batch_idx):
        """ Create and save figure with 2 images: original and prediction """
        fig = self.get_fig(raw_img, fancy_unet_pred, segformer_pred, mask, pred)
        fig.savefig(Path(self.logger.log_dir) / f'{self.name}_{phase}_epoch{self.current_epoch}_batch{batch_idx}.png')
        plt.close(fig)

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad(set_to_none=True)

    def configure_callbacks(self):
        return [
            pl.callbacks.ModelCheckpoint(
                monitor='val/dice_score',
                dirpath=self.logger.log_dir,
                save_top_k=3,
                mode='max',
            ),
        ]


class FancyUNetModel(BaseModel):
    """ Model and parameters for using fancy_unet as backbone """

    def __init__(self):
        super().__init__(FancyUnet(
            in_channels=3,
            inter_channels=48,
            height=5,
            width=2,
            class_num=1
        ))

        self.batch_size = 1
        self.accumulate_grad_batches = 16

        self.name = 'fancy_unet'

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, eps=1e-7, weight_decay=1e-4)
        return optimizer

    def forward(self, x):
        return self.internal(x)


class SegformerModel(BaseModel):
    """ Model and parameters for using segformer as backbone """

    def __init__(self):
        super().__init__(segformer.get_model())

        self.batch_size = 4
        self.accumulate_grad_batches = 4

        self.name = 'segformer'

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, eps=1e-7, weight_decay=1e-4)
        return optimizer

    def forward(self, x):
        x = self.internal(x)[0]
        x = torch.functional.F.interpolate(x, size=(384, 384), mode='bilinear')  # TODO move var out
        return x


def main():
    """ Entry point for the program.
    Parses command line arguments and runs the appropriate function(s), including:
    - Training the model
    - Testing the model
    - Predicting with the model
    """
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=str, default=None)
    ap.add_argument('--model', type=str)
    ap.add_argument('--train', action='store_true', default=False)
    ap.add_argument('--test', action='store_true', default=False)
    ap.add_argument('--predict', action='store_true', default=False)
    args = ap.parse_args()
    print(args)
    if torch.cuda.is_available():
        print('Using CUDA')

    # dataset.main()

    if args.model == 'fancy_unet':
        Model = FancyUNetModel
    elif args.model == 'segformer':
        Model = SegformerModel
    else:
        raise ValueError(f'Unknown model: {args.model}')

    if args.checkpoint is not None:
        model = Model.load_from_checkpoint(args.checkpoint)
    else:
        model = Model()

    dm = CariesDataModule(model.batch_size, 1)

    trainer = pl.Trainer(
        log_every_n_steps=1,  # optimizer steps!
        max_epochs=-1,
        deterministic=False,
        accumulate_grad_batches=model.accumulate_grad_batches,
        reload_dataloaders_every_n_epochs=1,
        logger=pl.loggers.TensorBoardLogger(segment_out_dir),
    )

    if args.train:
        trainer.fit(model, dm)
    if args.test:
        trainer.test(model, dm)


if __name__ == '__main__':
    main()
