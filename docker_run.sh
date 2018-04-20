#!/usr/bin/env bash
if [ $# -eq 0 ]
    then
        echo "running local image"
        docker run -it \
        -p 5901:5901 \
        -p 6901:6901 \
        -p 8097:8097 \
        baby-ai-game:safety_envelope \
        bash
elif [ $1 = "hub" ]
    then
        echo "updating from dockerhub"
        docker pull pmallozzi/baby-ai-game:safety_envelope
        echo "running image"
        docker run -it \
	    -p 5901:5901 \
	    -p 6901:6901 \
	    -p 8097:8097 \
	    pmallozzi/baby-ai-game:safety_envelope \
	    bash
else
   echo "Unknown argument. Write 'hub'."
fi
