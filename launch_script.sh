#!/usr/bin/env bash

configuration_file=${1:-"main.json"}

echo "working directory"
echo $PWD

if [ $# -eq 0 ]
  then
    echo "using default configuration file: $configuration_file"
else
    echo "...updating selected configuration file..."
    cd ./configurations
    yes | cp -rf $configuration_file "main.json"
fi

# Use virtual environment if exists
if [ -d "venv" ]; then
  echo "...activating python venv..."
  source ./venv/bin/activate
fi


# echo "...launch visdom server in the background..."
# python3 -m visdom.server &

echo "...setting up python environment..."
PYTHONPATH=../gym-minigrid/:../gym-minigrid/gym_minigrid/:./:$PYTHONPATH
export PYTHONPATH

/bin/bash

#echo "...launching the training..."
#python3 ./pytorch_rl/main.py
