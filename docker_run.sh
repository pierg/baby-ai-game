#!/usr/bin/env bash


envname=$1

if [ $# -eq 0 ]
  then
    echo "Specify the local path of the baby-ai-game folder"
    exit 1
fi


docker run -it \
	-p 5901:5901 \
	-p 6901:6901 \
	-v $1:/headless/baby-ai-game \
	baby-ai-game \
	bash