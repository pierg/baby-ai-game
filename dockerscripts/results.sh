#!/usr/bin/env bash

sudo docker exec -t $(sudo docker ps -lq) mkdir -p /headless/baby-ai-game/results

echo "Launching plot scripts in all the running containers..."
sudo docker exec -t $(sudo docker ps -lq) python3 plotResult.py
echo "...done"
echo ""


echo "Extracting the plot folder from all the running containers..."
for pid in `sudo docker ps -q`; do
	sudo docker cp $pid:/headless/baby-ai-game/results ~/results
done
echo "...done"
echo ""

exit
