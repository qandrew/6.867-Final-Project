#!/bin/bash

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

# This is a shell script, but it's recommended that you run the commands one by
# one by copying and pasting into the shell.

# Assumes local/data_files/data_prep.py has been run.

numjobs=80

mkdir -p data/train_fbank
mkdir -p data/dev_fbank
mkdir -p data/test_fbank

cp data/train/text data/train/spk2utt data/train/wav.scp data/train/utt2spk data/train_fbank
cp data/dev/text data/dev/spk2utt data/dev/wav.scp data/dev/utt2spk data/dev_fbank
cp data/test/text data/test/spk2utt data/test/wav.scp data/test/utt2spk data/test_fbank

# Now make filter bank features.
# fbankdir should be some place with a largish disk where you
# want to store the features.
fbankdir=exp/fbank
for x in train_fbank dev_fbank test_fbank; do
  steps/make_fbank.sh --cmd "$train_cmd" --nj $numjobs \
    data/$x exp/make_fbank/$x $fbankdir || exit 1;
  utils/fix_data_dir.sh data/$x
  steps/compute_cmvn_stats.sh data/$x exp/make_fbank/$x $fbankdir || exit 1;
done

exit 0;
