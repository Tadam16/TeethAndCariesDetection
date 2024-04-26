from transformers import SegformerForSemanticSegmentation

import torch


def get_model():
    """ Returns a Segformer model with 3 output classes (layers) """
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b2-finetuned-ade-512-512",
        return_dict=False,
        num_labels=1,
        ignore_mismatched_sizes=True,
        cache_dir='/cache/segformer_cache'
    )

    return model


if __name__ == "__main__":
    """ Test the model input-output format """
    img = torch.zeros(1, 3, 384, 384)
    model = get_model()
    out = model(img)
    print(model)
    print(out)
