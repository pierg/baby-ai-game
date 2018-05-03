#!/usr/bin/env bash

configuration_file=${1:-"main.json"}

if [ $# -eq 0 ]
  then
    echo "using default configuration file: $configuration_file"
fi


# Pull latest changes in the repositories
echo "...updating repositories..."
git pull
cd ..
cd gym-minigrid
git pull
cd ..

cd ./baby-ai-game

# Use virtual environment if exists
if [ -d "venv" ]; then
  echo "...activating python venv..."
  source ./venv/bin/activate
fi


echo "...launch visdom server in the background..."
python3 -m visdom.server &


echo "...configuring PYTHONPATH..."
PYTHONPATH=../gym-minigrid/:../gym-minigrid/gym_minigrid/:./:$PYTHONPATH
export PYTHONPATH
