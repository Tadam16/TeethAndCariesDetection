source ../declare.sh

docker run \
  --gpus all \
  -it \
  -v "${CACHE_DIR}:/cache" \
  -v "${CODE_DIR}:/code" \
  -v "${DATA_DIR}:/data" \
  -v "${OUT_DIR}:/out" \
  melytanulas:latest \
  /bin/sh -c "/opt/conda/bin/python3 /code/presegment/trainer.py --predict --model fancy_unet --checkpoint '/out/presegment/lightning_logs/fancy_unet/epoch=17-step=1350.ckpt' "
