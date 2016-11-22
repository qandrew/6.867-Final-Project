#!/bin/bash
# This script iterates through all subdirectories and repairs all .wav files
# using the script sph2pipe. Replaces the original files with the repaired files 
# of the same name.

clear 
echo "Repairing broken .wav files ${PWD%/*}"

find $(dir)/data -type f -iname '*.wav' -print0 |
while IFS= read -r -d '' f; 
do 
	echo "Cleaning $f" 
	sh  ~/${PWD%/*}/scripttest/sph2pipe.sh -f wav $f $f
done
$SHELL
