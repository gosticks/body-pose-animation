#!/bin/bash

mkdir models
rm -rf ./tmp
mkdir -p ./tmp/smpl

echo "Downloading SMPL model"
wget -O ./tmp/smpl.source.zip http://smplify.is.tue.mpg.de/main/download1
bsdtar -x -f ./tmp/smpl.source.zip -C ./tmp/smpl
mkdir -p ./models/smpl
cp ./tmp/smpl/smplify_public/code/models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl ./models/smpl/SMPL_NEUTRAL.pkl
cp ./tmp/smpl/smplify_public/code/models/regressors_locked_normalized_female.npz ./models/smpl/SMPL_FEMALE.npz
cp ./tmp/smpl/smplify_public/code/models/regressors_locked_normalized_male.npz ./models/smpl/SMPL_MALE.npz
