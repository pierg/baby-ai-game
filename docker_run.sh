#!/usr/bin/env bash
if [ $# -eq 0 ]
  then
    docker run -it \
	-p 5901:5901 \
	-p 6901:6901 \
	-p 8097:8097 \
	baby-ai-game:safety_envelope \
	bash
fi
else
    if [ $1 == "hub" ]
    then
        docker run -it \
	    -p 5901:5901 \
	    -p 6901:6901 \
	    -p 8097:8097 \
	    pmallozzi/baby-ai-game:safety_envelope \
	    bash
    fi