from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from flask import Flask, request, send_file
from torchvision.utils import save_image

import presegment.dataset
import presegment.trainer
import segment.dataset
import segment.trainer


def print(*args, **kwargs):
    if parsed.debug:
        __builtins__.print(*args, **kwargs, flush=True)


app = Flask(__name__)


def get_model(phase, model_name, checkpoint_path):
    if phase == 'presegment':
        library = presegment
    elif phase == 'segment':
        library = segment
    else:
        raise ValueError('Phase not found')

    if model_name == 'fancy_unet':
        Model = library.trainer.FancyUNetModel
    elif model_name == 'segformer':
        Model = library.trainer.SegformerModel
    else:
        raise ValueError('Model type not found')

    if checkpoint_path is not None:
        model = Model.load_from_checkpoint(f'/out/{phase}/lightning_logs/{checkpoint_path}')
        model.eval()
    else:
        raise ValueError('Checkpoint not found')

    return model


def presegment_segformer_predict(batch):
    presegment_model_segformer = get_model(
        'presegment',
        'segformer',
        'segformer_10_epochs/epoch=9-step=750.ckpt'
    )

    batch = presegment_model_segformer.transfer_batch_to_device(batch, presegment_model_segformer.device, ...)
    _, pred, _, _ = presegment_model_segformer.predict_step(batch, ...)
    np.save(
        str('/workspace/segformer_pred.npy'),
        pred.detach().cpu().numpy()[0, 0]
    )


def presegment_fancy_unet_predict(batch):
    presegment_model_fancy_unet = get_model(
        'presegment',
        'fancy_unet',
        'fancy_unet_10_epochs/epoch=9-step=750.ckpt'
    )

    batch = presegment_model_fancy_unet.transfer_batch_to_device(batch, presegment_model_fancy_unet.device, ...)
    _, pred, _, _ = presegment_model_fancy_unet.predict_step(batch, ...)
    np.save(
        str('/workspace/fancy_unet_pred.npy'),
        pred.detach().cpu().numpy()[0, 0]
    )


def segment_predict(batch):
    segment_model = get_model(
        'segment',
        'fancy_unet',
        'middle_downscaled/epoch=117-step=3304.ckpt'
    )

    batch = segment_model.transfer_batch_to_device(batch, segment_model.device, ...)
    raw_img, fancy_unet_pred, segformer_pred, img, mask = batch
    _, pred, _, _, _, _ = segment_model.predict_step(batch, ...)

    result = torch.stack([img[0, 0]] * 3, dim=-1) * 0.7
    result[..., 0] += pred[0, 0] * 0.3

    save_image(result, '/workspace/output.jpg')




@app.route('/predict', methods=['POST'])
def handle_request():
    # input_path = '/workspace/input.png'

    if 'image' not in request.files:
        return 'No image found', 400

    # Load image

    input_path = request.files['image']

    # Presegment

    presegment_dataset = presegment.dataset.TeethDataset(pd.DataFrame({
        'image': [input_path],
        'mask': [None],
        'id': [0],
    }))

    raw_img, img, mask, id_ = presegment_dataset[0]
    batch = raw_img[None], img[None], mask[None], id_[None]
    presegment_fancy_unet_predict(batch)
    presegment_segformer_predict(batch)

    # Segment

    segment_dataset = segment.dataset.CariesDataset(pd.DataFrame({
        'image': [input_path],
        'fancy_unet_pred': ['/workspace/fancy_unet_pred.npy'],
        'mask': [None],
        'bbox': [None],
        'segformer_pred': ['/workspace/segformer_pred.npy'],
        'id': [0],
    }),
        size=segment.trainer.image_size)

    raw_img, fancy_unet_pred, segformer_pred, img, mask = segment_dataset[0]
    batch = raw_img[None], fancy_unet_pred[None], segformer_pred[None], img[None], mask[None]
    segment_predict(batch)

    # Return result

    return send_file('/workspace/output.jpg', mimetype='image/jpg')


def main():
    global parsed
    ap = ArgumentParser()
    ap.add_argument("-d", "--debug", required=False, help="Debug mode", default=False)
    parsed = ap.parse_args()

    Path('/workspace').mkdir(exist_ok=True)

    app.run(host='0.0.0.0', port=6006, debug=parsed.debug)


if __name__ == '__main__':
    main()
