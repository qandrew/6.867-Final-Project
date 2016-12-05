#!/bin/bash
# This script iterates through all subdirectories and moves all single utterances to a new folder

clear 
echo "Randomly selecting test data with probability 1/4 from ${PWD%/}"

find $(dir/single_utterance) -type f -iname '*.txt' -print0 |
while IFS= read -r -d '' f; 
do 
	file=$(basename $f)
	if [ $((1 + RANDOM % 40)) -lt 10]; then
		mv $f $PWD/single_utterance/test
		echo "Copying $file to testing data folder"  
	else 
		mv $f $PWD/single_utterance/train
		echo "Copying $file to training data folder"  
	fi
done
$SHELL
