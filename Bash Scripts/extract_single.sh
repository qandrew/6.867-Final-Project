#!/bin/bash
# This script iterates through all subdirectories and moves all single utterances to a new folder

clear 
echo "Selecting Single Utterances in Subfolders of ${PWD%/}"

find $(dir/data) -type f -iname '*.txt' -print0 |
while IFS= read -r -d '' f; 
do 
	file=$(basename $f)
	echo "Cleaning $f with length ${#file}" 
	# If the file is a single utterance, it is of the form xx_#x.wav; length 9
	
	if [ "${#file}" -eq 9 ]; then
		cp $f $PWD/single_utterance
		echo "Copying $file to new folder"  
	fi
done
$SHELL
