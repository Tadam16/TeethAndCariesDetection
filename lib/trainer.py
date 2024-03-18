from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.io import savemat

import segformer
from dataset import TeethDataModule
import torch.utils.data
from fancy_unet import Unet as FancyUnet
from path_util import out_dir

import torchmetrics
import lightning.pytorch as pl


class BaseModel(pl.LightningModule):
    """ Common settings for the two models """

    def __init__(self, internal):
        super().__init__()

        self.internal = internal

        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.dice_score_fn = torchmetrics.Dice(zero_division=1)

        self.dice_frequency = 32  # MUST be min accumulate_grad_batches and SHOULD be equal

    def forward(self, x):
        return self.internal(x)

    def predict_step(self, batch, batch_idx):
        raw_img, img, mask = batch
        pred_raw = self(img)
        pred = torch.sigmoid(pred_raw)
        return raw_img, pred, mask

    def training_step(self, batch, batch_idx):
        """ Train, occasionally calculate metrics, occasionally show figures"""
        raw_img, img, mask = batch
        pred_raw = self(img)
        loss = self.loss_fn(pred_raw, mask.float())
        self.log(f'train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        if batch_idx % self.dice_frequency == 0:
            pred = torch.sigmoid(pred_raw)
            dice_score = self.dice_score_fn(pred, mask)
            self.log(f'train/dice_score', dice_score, on_step=True, on_epoch=True, prog_bar=True)

        if batch_idx % (self.dice_frequency * 10) == 0:
            self.show_fig('train', raw_img, mask, pred, batch_idx)

        return loss

    def validation_step(self, batch, batch_idx):
        """ Calculate metrics, show figures for 10 fixed images"""
        raw_img, img, mask = batch
        pred_raw = self(img)
        loss = self.loss_fn(pred_raw, mask.float())
        self.log(f'train/loss', loss, on_epoch=True)

        pred = torch.sigmoid(pred_raw)
        dice_score = self.dice_score_fn(pred, mask)
        self.log(f'val/dice_score', dice_score, on_epoch=True)

        if batch_idx <= 10:
            self.show_fig('val', raw_img, mask, pred, batch_idx)

    def test_step(self, batch, batch_idx):
        """ Calculate metrics including custom metric, show figures for 10 fixed images"""
        raw_img, img, mask = batch
        pred_raw = self(img)
        loss = self.loss_fn(pred_raw, mask.float())
        self.log(f'test/loss', loss, on_epoch=True)

        pred = torch.sigmoid(pred_raw)
        dice_score = self.dice_score_fn(pred, mask)
        self.log(f'test/dice_score', dice_score, on_epoch=True)

        if batch_idx <= 10:
            self.show_fig('test', raw_img, mask, pred, batch_idx)

        return loss

    def show_fig(self, phase, img, mask, pred, batch_idx):
        """ Create and save figure with 2 images: original and prediction """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        img1 = np.stack([img.detach().cpu().numpy()[0, 0]] * 3, axis=-1) * 0.7
        pred = pred.detach().cpu().numpy()
        img1[..., 0] += pred[0, 0] * 0.3
        ax1.imshow(img1)
        img2 = np.stack([img.detach().cpu().numpy()[0, 0]] * 3, axis=-1) * 0.7
        img2[..., 0] += mask.detach().cpu().numpy()[0, 0] * 0.3
        ax2.imshow(img2)
        fig.savefig(Path(self.logger.log_dir) / f'{self.name}_{phase}_epoch{self.current_epoch}_{batch_idx}.png')

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
            in_channels=1,
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
        x = torch.cat([x] * 3, dim=1)
        x = self.internal(x)[0]
        x = torch.functional.F.interpolate(x, size=(384, 384), mode='bilinear')
        return x


def main(args):
    """ Create or load model, create data module and trainer, run training and testing """
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

    dm = TeethDataModule(model.batch_size, 3 if args.colab else 1)

    trainer = pl.Trainer(
        log_every_n_steps=1,  # optimizer steps!
        max_epochs=10,
        deterministic=False,
        accumulate_grad_batches=model.accumulate_grad_batches,
        reload_dataloaders_every_n_epochs=1,
        logger=pl.loggers.TensorBoardLogger(out_dir),
    )

    if args.train:
        trainer.fit(model, dm)
    if args.test:
        trainer.test(model, dm)

    if args.predict:
        dice_score = torchmetrics.Dice(zero_division=1)
        for loader_name, loader in {
            # 'val': dm.val_dataloader(batch_size=1),
            # 'test': dm.test_dataloader(batch_size=1),
            'predict': dm.predict_dataloader(batch_size=1)
        }.items():
            (out_dir / 'predictions' / model.name / loader_name).mkdir(parents=True, exist_ok=True)
            for i, (raw_img, pred, mask) in enumerate(
                    trainer.predict(model, dataloaders=[loader], return_predictions=True)
            ):
                savemat(str(out_dir / 'predictions' / model.name / loader_name / f'{i}.mat'), {
                    'raw_img': raw_img.numpy(),
                    'pred': pred.numpy(),
                    'mask': mask.float().numpy(),
                    'dice_score': dice_score(pred, mask).item(),
                })
                model.show_fig('predict', raw_img, mask, pred, i)
        # TODO save in numpy as well
        # TODO map back to original paths
        # TODO support ensemble mode
