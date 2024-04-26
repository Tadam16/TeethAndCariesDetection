source declare.sh

docker run \
  --gpus all \
  -it \
  -p 6006:6006 \
  -v "${CACHE_DIR}:/cache" \
  -v "${CODE_DIR}:/code" \
  -v "${DATA_DIR}:/data" \
  -v "${OUT_DIR}:/out" \
  melytanulas:latest \
  /bin/bash -c "tensorboard --logdir=/out --port=6006 --bind_all"
