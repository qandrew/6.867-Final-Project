#!/bin/bash

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

# Assumes local/data_files/data_prep.py and run_make_fbank_feats.sh have been run.
# This script splits kaldi archives of features into individual text files,
# where each file contains the features for one utterance. Useful when you want
# to use kaldi to extract features to be used by some other toolkit.

nj=80
cmd=utils/hostd.pl

mkdir -p data/train_fbank
mkdir -p data/dev_fbank
mkdir -p data/test_fbank

cp data/train/text data/train/spk2utt data/train/wav.scp data/train/utt2spk data/train_fbank
cp data/dev/text data/dev/spk2utt data/dev/wav.scp data/dev/utt2spk data/dev_fbank
cp data/test/text data/test/spk2utt data/test/wav.scp data/test/utt2spk data/test_fbank

# Now make filter bank features.
# fbankdir should be some place with a largish disk where you
# want to store the features.
for set in train_fbank dev_fbank test_fbank; do
  ./utils/split_data.sh --per-utt data/$set $nj
  for i in $(seq 1 $nj); do
	  srcfeats=data/$set/split$nj/$i/feats.scp
	  destfeats=data/$set/split$nj/$i/feats_by_utt.scp
	  temptags=data/$set/split$nj/$i/tags.temp
	  temparknums=data/$set/split$nj/$i/arknums.temp
	  dirs2make=data/$set/split$nj/$i/dirs2make.temp
	  cut -d '.' -f2 $srcfeats > $temparknums
	  awk '{print "exp/fbank_by_utt/" $1}' < $temparknums | sort | uniq > $dirs2make
	  awk '{print $1}' < $srcfeats > $temptags
	  paste -d ' ' $temptags $temparknums | awk '{print $1 " exp/fbank_by_utt/" $2 "/" $1 ".txt" }' > $destfeats
	  cat $dirs2make | xargs -I{} mkdir -p {}
	  rm -f $temptags
	  rm -f $temparknums
	  rm -f $dirs2make
  done

  srcfeats=data/$set/feats.scp
  destfeats=data/$set/feats_by_utt.scp
  temptags=data/$set/tags.temp
  temparknums=data/$set/arknums.temp
  dirs2make=data/$set/dirs2make.temp
  cut -d '.' -f2 $srcfeats > $temparknums
  awk '{print "exp/fbank_by_utt/" $1}' < $temparknums | sort | uniq > $dirs2make
  awk '{print $1}' < $srcfeats > $temptags
  paste -d ' ' $temptags $temparknums | awk '{print $1 " exp/fbank_by_utt/" $2 "/" $1 ".txt" }' > $destfeats
  cat $dirs2make | xargs -I{} mkdir -p {}
  rm -f $temptags
  rm -f $temparknums
  rm -f $dirs2make

  logdir=data/$set/splitlog
  mkdir -p $logdir
  $cmd JOB=1:$nj $logdir/split_fbank_${set}.JOB.log \
    copy-feats --verbose=2 --binary=false scp:data/$set/split$nj/JOB/feats.scp scp,t:data/$set/split$nj/JOB/feats_by_utt.scp || exit 1;
done

exit 0;
