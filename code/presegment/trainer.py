import argparse
from pathlib import Path
# import cv2

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

import common.segformer as segformer
from dataset import TeethDataModule
import torch.utils.data
from common.fancy_unet import Unet as FancyUnet
from common.path_util import presegment_out_dir, data_dir, rebase

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
        raw_img, img, mask, id_ = batch
        pred_raw = self(img)
        pred = torch.sigmoid(pred_raw)
        return raw_img, pred, mask, id_

    def training_step(self, batch, batch_idx):
        """ Train, occasionally calculate metrics, occasionally show figures"""
        raw_img, img, mask, id_ = batch
        pred_raw = self(img)
        loss = self.loss_fn(pred_raw, mask.float())
        self.log(f'train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        if batch_idx % self.dice_frequency == 0:
            pred = torch.sigmoid(pred_raw)
            dice_score = self.dice_score_fn(pred, mask)
            self.log(f'train/dice_score', dice_score, on_step=True, on_epoch=True, prog_bar=True)

            if batch_idx % (self.dice_frequency * 10) == 0:
                self.show_example('train', raw_img, mask, pred, batch_idx)

        return loss

    def validation_step(self, batch, batch_idx):
        """ Calculate metrics, show figures for 10 fixed images"""
        raw_img, img, mask, id_ = batch
        pred_raw = self(img)
        loss = self.loss_fn(pred_raw, mask.float())
        self.log(f'val/loss', loss, on_epoch=True)

        pred = torch.sigmoid(pred_raw)
        dice_score = self.dice_score_fn(pred, mask)
        self.log(f'val/dice_score', dice_score, on_epoch=True)

        if batch_idx <= 10:
            self.show_example('val', raw_img, mask, pred, batch_idx)

    def test_step(self, batch, batch_idx):
        """ Calculate metrics including custom metric, show figures for 10 fixed images"""
        raw_img, img, mask, id_ = batch
        pred_raw = self(img)
        loss = self.loss_fn(pred_raw, mask.float())
        self.log(f'test/loss', loss, on_epoch=True)

        pred = torch.sigmoid(pred_raw)
        dice_score = self.dice_score_fn(pred, mask)
        self.log(f'test/dice_score', dice_score, on_epoch=True)

        if batch_idx <= 10:
            self.show_example('test', raw_img, mask, pred, batch_idx)

        return loss

    @staticmethod
    def get_fig(raw_img, mask, pred):
        """ Create and save figure with 2 images: prediction adn ground truth"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        img1 = np.stack([raw_img.detach().cpu().numpy()[0, 0]] * 3, axis=-1) * 0.7
        pred = pred.detach().cpu().numpy()
        img1[..., 0] += pred[0, 0] * 0.3
        ax1.imshow(img1)
        ax1.set_title(f'Prediction')
        img2 = np.stack([raw_img.detach().cpu().numpy()[0, 0]] * 3, axis=-1) * 0.7
        img2[..., 0] += mask.detach().cpu().numpy()[0, 0] * 0.3
        ax2.imshow(img2)
        ax2.set_title(f'Ground truth')
        return fig

    def show_example(self, phase, raw_img, mask, pred, batch_idx):
        """ Create and save figure with 2 images: original and prediction """
        fig = self.get_fig(raw_img, mask, pred)
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
    ap.add_argument('--model', type=str, default='fancy_unet')
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

    dm = TeethDataModule(model.batch_size, 1)

    trainer = pl.Trainer(
        log_every_n_steps=1,  # optimizer steps!
        max_epochs=20,
        deterministic=False,
        accumulate_grad_batches=model.accumulate_grad_batches,
        reload_dataloaders_every_n_epochs=1,
        logger=pl.loggers.TensorBoardLogger(presegment_out_dir),
    )

    if args.train:
        trainer.fit(model, dm)
    if args.test:
        trainer.test(model, dm)

    if args.predict:
        batch_size = 2
        for loader_name, loader in {
            'val': dm.val_dataloader(batch_size=batch_size),
            'test': dm.test_dataloader(batch_size=batch_size),
            'predict': dm.predict_dataloader(batch_size=batch_size),
        }.items():
            print('Predicting:', loader_name)
            for batch in tqdm(loader):
                raw_img, img, mask, id_ = batch

                should_predict = False
                for i in range(len(id_)):
                    path = dm.paths[dm.paths['id'] == int(id_[i])]['image'].iloc[0]

                    new_path = rebase(data_dir, presegment_out_dir / 'predictions' / model.name / loader_name, path)

                    if not (new_path.parent / f'{new_path.stem}.npy').exists():
                        should_predict = True

                if should_predict:
                    batch = model.transfer_batch_to_device(batch, model.device, ...)
                    _, pred, _, _ = model.predict_step(batch, ...)
                    pred = pred.detach().cpu()

                    for i in range(len(id_)):
                        path = dm.paths[dm.paths['id'] == int(id_[i])]['image'].iloc[0]

                        new_path = rebase(data_dir, presegment_out_dir / 'predictions' / model.name / loader_name, path)
                        new_path.parent.mkdir(parents=True, exist_ok=True)

                        # TODO try with opencv
                        # TODO add options
                        # TODO coordinate between output formats
                        # save_img = pred.numpy()[i]
                        # save_img = np.stack([save_img] * 3, axis=-1)
                        # save_img = save_img.transpose(2, 1, 0) * 255.0
                        # Save as npz
                        # cv2.imwrite(
                        #     str(new_path.parent / f'{new_path.stem}.png'),
                        #     save_img,
                        # )
                        np.save(
                            str(new_path.parent / f'{new_path.stem}.npy'),
                            pred.numpy()[i, 0],
                        )

                        # # Save as picture
                        # fig = model.get_fig(raw_img, mask, pred)
                        # fig.savefig(new_path.parent / f'{new_path.stem}.png')
                        # plt.close(fig)


if __name__ == '__main__':
    main()
