#!/usr/bin/env bash
if [ $# -eq 0 ]
    then
        docker run -it \
        -p 5901:5901 \
        -p 6901:6901 \
        -p 8097:8097 \
        --entrypoint setup_script.sh \
        pmallozzi/baby-ai-game:safety_envelope
else
   echo "running local image with configuration file: $1"
        docker run -it \
            -p 5901:5901 \
            -p 6901:6901 \
            -p 8097:8097 \
            pmallozzi/baby-ai-game:safety_envelope $1
fi
