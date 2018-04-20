#!/usr/bin/env bash

configuration_file=${1:-"no_blocker.json"}

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

echo "...updating selected configuration file..."
cd ./baby-ai-game/configurations
yes | cp -rf $configuration_file "main.json"

cd ..

# Use virtual environment if exists
if [ -d "venv" ]; then
  echo "...activating python venv..."
  source ./venv/bin/activate
fi

echo "...start training..."
PYTHONPATH=../gym-minigrid/:../gym-minigrid/gym_minigrid/:./:$PYTHONPATH
export PYTHONPATH
python3 ./pytorch_rl/main.py
