#!/usr/bin/env bash

# Sets the main.json as default, if the -t is specifed
# it will use that as config file.
configuration_file="main.json"
start_training=0
seed=0
seed_val=0
no_monitor=0

OPTIND=1         # Reset in case getopts has been used previously in the shell.

while getopts t:rs:m opt; do
    case ${opt} in
        r)
            random=1
            start_training=1
            echo "random"
            ;;
        t)
            configuration_file=$OPTARG
            start_training=1
            echo "config"
            ;;
        s)
            seed_val=$OPTARG
            seed=1
            echo $OPTARG
            echo "seed set"
            ;;
        m)
            no_monitor=1
            echo "no monitor"
            ;;
    esac
done
shift $((OPTIND -1))

if [ ${random} ]
    then
        echo "...creating a random environment..."
        echo "...creating environment with grid_size 6, number of water tiles 2, max block size 1, with default reward config, without a monitor"
        if [ ${seed} -eq 1 ]
            then
                if [ ${no_monitor} -eq 1 ]
                    then
                        echo "Seed and no_monitor provided, creating from seed without monitor"
                        configuration_file=`python3 env_generator.py --grid_size 6 --number_of_water_tiles 2 --max_block_size 1 --rewards_file "configurations/rewards/default.json" --no-monitor --seed ${seed_val}`
                    else
                        echo "Seed with monitor provied, creating from seed wih a montior"
                        configuration_file=`python3 env_generator.py --grid_size 6 --number_of_water_tiles 2 --max_block_size 1 --rewards_file "configurations/rewards/default.json" --seed ${seed_val}`
                fi
            else
                if [ ${no_monitor} -eq 1 ]
                    then
                        echo "No seed was provided but no_monitor was, creating random without a monitor"
                        configuration_file=`python3 env_generator.py --grid_size 6 --number_of_water_tiles 2 --max_block_size 1 --rewards_file "configurations/rewards/default.json" --no-monitor`
                    else
                        configuration_file=`python3 env_generator.py --grid_size 6 --number_of_water_tiles 2 --max_block_size 1 --rewards_file "configurations/rewards/default.json"`
                fi
        fi
    else
        configuration_file=${1:-"main.json"}
fi

echo "...environment name is..."
echo $configuration_file


if [ $configuration_file -eq "main.json" ]
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
PYTHONPATH=../gym-minigrid/:../gym-minigrid/gym_minigrid/:./configurations:./:$PYTHONPATH
export PYTHONPATH


if [ $start_training -eq 1 ]
  then
    echo "...launching the training..."
    python3 ./pytorch_rl/main.py
else
    echo "environment ready!"
    /bin/bash
fi


# echo "...launch visdom server in the background..."
# python3 -m visdom.server &
