#!/usr/bin/env bash
docker run -it \
	-p 5901:5901 \
	-p 6901:6901 \
	-p 8097:8097 \
	baby-ai-minigrid \
	bash