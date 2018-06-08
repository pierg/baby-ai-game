#!/usr/bin/env bash
# Generate a random Environment and screenshot it

# Generate a random Environment
configuration_file=`python3 env_generator.py --environment_file "default" --rewards_file "default"`

# Generate the name of the copy with and without path
copyName="copyOf$configuration_file"
copyName=`basename $copyName .json`
copyName="randoms/$copyName"
copyPath="configurations/randoms/copyOf$configuration_file"
configuration_file="configurations/randoms/$configuration_file"

# Change number of process and rendering in order to take the screen
sed 's/.num_processes.: 48/"num_processes": 1/g; s/.rendering.: false/"rendering": true/g' $configuration_file > $copyPath

# Screenshot it
#python3 screenHelper.py $copyName


