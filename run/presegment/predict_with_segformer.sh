source ../declare.sh

docker run \
  --gpus all \
  -it \
  -v "${CACHE_DIR}:/cache" \
  -v "${CODE_DIR}:/code" \
  -v "${DATA_DIR}:/data" \
  -v "${OUT_DIR}:/out" \
  melytanulas:latest \
  /bin/sh -c "/opt/conda/bin/python3 /code/presegment/trainer.py --predict --model segformer --checkpoint '/out/presegment/lightning_logs/segformer_10_epochs/epoch=9-step=750.ckpt' "