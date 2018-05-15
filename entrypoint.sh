#!/usr/bin/env bash

# Pull latest changes in the repositories
echo "...updating repositories..."
git pull
cd ..
cd gym-minigrid
git pull
cd ..

cd baby-ai-game

if [ $# -eq 0 ]
  then
    source launch_script.sh
else
    source launch_script.sh $1
fi

