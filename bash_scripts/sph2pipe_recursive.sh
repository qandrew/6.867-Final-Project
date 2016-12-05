#!/bin/bash
# My first script

echo "Starting to Repair Broken Wav Files"

#loop through all subdirectories of data from current directory 

find ./data -type f -iname '*.wav' -print0 | 
while IFS= read -r -d '' f;
do 
	echo "Cleaning $f"
	$PWD/sph2pipe.sh -f wav $f $f"-"
done
find ./data -type f -iname '*.wav-' -print0 | 
#find ./data -type f -iname '*.wav' -print0 | 
while IFS= read -r -d '' f;
do 
	echo "Renaming $f"
	tempf="${f%-*}"
	mv $f $tempf
	echo $tempf
done
#for subdir in ~/data/*;
#  do 
#     [ -d $subdir ] && cd "$subdir" && echo "Entering into $subdir and looking for wav files..."
     # Find all files with the .wav extension
#     for f in "$^(.(?!yo_))*.wav"; do
         # do some stuff here with "$f"
#         echo "Repairing $f ..."
#	 $PWD/sph2pipe.sh -f wav $f "yo_"$f
#    done;
# done;
echo -e "\n\nDONE"

