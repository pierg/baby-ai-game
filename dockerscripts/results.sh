#!/usr/bin/env bash

echo "Launching plot scripts in all the running containers..."
sudo docker exec -t $(sudo docker ps -lq) python3 plotResults
echo "...done"
echo ""


echo "Extracting the plot folder from all the running containers..."
for pid in `sudo docker ps -q`; do
	sudo docker cp $pid:results ~/results
done
echo "...done"
echo ""

exit
