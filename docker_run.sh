#!/usr/bin/env bash
if [ $# -eq 0 ]
    then
        echo "running local image with default configuration"
        docker run -it \
        -p 5901:5901 \
        -p 6901:6901 \
        -p 8097:8097 \
        baby-ai-game:safety_envelope
else
   echo "running local image with configuration file: $1"
        docker run -it \
            -p 5901:5901 \
            -p 6901:6901 \
            -p 8097:8097 \
            baby-ai-game:safety_envelope $1
fi
