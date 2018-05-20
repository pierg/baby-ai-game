#!/usr/bin/env bash

# Sets the main.json as default, if the -t is specifed
# it will use that as config file.
configuration_file="main.json"

while getopts ":tr" opt; do
    case ${opt} in
        r)
            random=1
            ;;
        t)
            configuration_file=${OPTARG}
            ;;
    esac
done
shift $((OPTIND -1))

if [ $random ]
    then
        echo "...creating a random environment..."
        echo "...creating environment with grid_size 6, number of water tiles 3, max block size 1, with default reward config"
        configuration_file=`python3 env_generator.py --grid_size 6 --number_of_water_tiles 3 --max_block_size 1 --rewards_file "configurations/rewards/default.json"`
    else
        configuration_file=${1:-"main.json"}
fi

echo "...environment name is..."
echo $configuration_file

if [ $# -eq 0 ]
  then
    echo "using default configuration file: $configuration_file"
else
    echo "...updating selected configuration file..."
    cd ./configurations
    yes | cp -rf $configuration_file "main.json"
    echo "using configuration file: $configuration_file"
fi

cd ..

# Use virtual environment if exists
if [ -d "venv" ]; then
  echo "...activating python venv..."
  source ./venv/bin/activate
fi

echo "...setting up python environment..."
PYTHONPATH=../gym-minigrid/:../gym-minigrid/gym_minigrid/:./:$PYTHONPATH
export PYTHONPATH

echo "...launching the training..."
echo $PWD
python ./pytorch_rl/main.py

# echo "...launch visdom server in the background..."
# python3 -m visdom.server &
