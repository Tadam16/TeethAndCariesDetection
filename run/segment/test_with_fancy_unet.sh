source ../declare.sh

docker run \
  --gpus all \
  -it \
  -v "${CACHE_DIR}:/cache" \
  -v "${CODE_DIR}:/code" \
  -v "${DATA_DIR}:/data" \
  -v "${OUT_DIR}:/out" \
  melytanulas:latest \
  /bin/sh -c "/opt/conda/bin/python3 /code/segment/trainer.py --test --model fancy_unet --checkpoint '/out/segment/lightning_logs/original_best_unet/epoch=122-step=615.ckpt' "
