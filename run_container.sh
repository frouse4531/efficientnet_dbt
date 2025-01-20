#!/usr/bin/env bash

docker run \
	--gpus all \
	--mount type=bind,source="$(pwd)",target=/efficientnet_dbt \
	--mount type=bind,source="/data",target=/data \
	--rm --ipc=host -it effdbt:latest
