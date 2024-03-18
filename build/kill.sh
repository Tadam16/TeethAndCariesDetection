# For easily killing all running containers
docker kill "$(docker ps -q)"
docker container prune -f