# For easily running the container

export DATA_DIR="/media/jbl/Secondary SSD/TeethAndCariesDetection/data/"
export OUT_DIR="/media/jbl/Secondary SSD/TeethAndCariesDetection/out/"
export CODE_DIR="/media/jbl/Secondary SSD/TeethAndCariesDetection/lib/"

docker run \
  --gpus all \
  -it \
  -v "${DATA_DIR}:/data" \
  -v "${OUT_DIR}:/out" \
  -v "${CODE_DIR}:/code" \
  melytanulas:latest \
  /bin/bash /code/startup.sh
