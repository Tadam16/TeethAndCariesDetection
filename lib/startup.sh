export PYTHONPATH=/code:$PYTHONPATH
#/opt/conda/bin/python3 /code/main.py --predict --model fancy_unet --checkpoint "/out/lightning_logs/fancy_unet_10_epochs/epoch=9-step=750.ckpt"
/opt/conda/bin/python3 /code/main.py --predict --model segformer --checkpoint "/out/lightning_logs/segformer_10_epochs/epoch=9-step=750.ckpt"