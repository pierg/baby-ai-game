#!/usr/bin/env bash

configuration_file=${1:-"default.json"}

if [ $# -eq 0 ]
  then
    echo "Using default configuration file"
fi


# Pull latest changes in the repositories
cd baby-ai-game
git pull
cd ..
cd gym-minigrid
git pull
cd ..

# Run simulation
cd ./baby-ai-game

source ./venv/bin/activate

PYTHONPATH=../gym-minigrid/:$PYTHONPATH
export PYTHONPATH
PYTHONPATH=../gym-minigrid/gym_minigrid/envs/:$PYTHONPATH
export PYTHONPATH
PYTHONPATH=./:$PYTHONPATH
export PYTHONPATH


python3 ./pytorch_rl/main.py --env-name MiniGrid-UnsafeEnvironment-6x6-v0 --no-vis --num-processes 48 --algo a2c
