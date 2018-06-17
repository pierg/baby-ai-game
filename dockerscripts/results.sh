#!/usr/bin/env bash

echo "Extracting the plot folder from all the running containers..."
for pid in `sudo docker ps -q`; do
	sudo docker cp $pid:/headless/baby-ai-game/evaluations/Unsafe-Random12x12-GOAP-1.csv ~/action_planning/results/$pid.csv
done
echo "...done"
echo ""

exit
