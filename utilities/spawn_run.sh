#!/usr/bin/env bash
docker pull pmallozzi/baby-ai-game:random_envs
seed=0
no_monitor=0
seed_val=0

OPTIND=1         # Reset in case getopts has been used previously in the shell.
while getopts s:m opt; do
    case ${opt} in
        s)
            seed_val=$OPTARG
            seed=1
            echo $OPTARG
            echo "seed set"
            ;;
        m)
            no_monitor=1
            echo "no monitor set"
            ;;
    esac
done
shift $((OPTIND -1))


if [ ${seed} -eq 1 ]
        then
                if [ ${no_monitor} -eq 1 ]
                        then
                                echo "Seed and no_monitor provided, creating from seed without monitor"
                                docker run -t -d -v /home/ubuntu/evaluations:/headless/baby-ai-game/evaluations pmallozzi/baby-ai-game:random_envs -r -s ${seed_val} -m
                        else
                                echo "Seed with monitor provied, creating from seed wih a montior"
                                docker run -t -d -v /home/ubuntu/evaluations:/headless/baby-ai-game/evaluations pmallozzi/baby-ai-game:random_envs -r -s ${seed_val}
                fi
        else
                if [ ${no_monitor} -eq 1 ]
                        then
                                echo "No seed was provided but no_monitor was, creating random without a monitor"
                                docker run -t -d -v /home/ubuntu/evaluations:/headless/baby-ai-game/evaluations pmallozzi/baby-ai-game:random_envs -r -m
                        else
                                echo "No seed and no_monitor were not provided, creating a random environment"
                                docker run -t -d -v /home/ubuntu/evaluations:/headless/baby-ai-game/evaluations pmallozzi/baby-ai-game:random_envs -r
                fi
fi
