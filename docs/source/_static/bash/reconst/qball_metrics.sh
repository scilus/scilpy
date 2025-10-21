#!/usr/bin/env bash
set -euo pipefail  # Will fail on error

# ==============
# How to run this script
#    1) Load the input data
#       https://scilpy.readthedocs.io/en/latest/documentation/getting_started.html
#    2) Call this script with
#    --->   bash btensor_scripts.sh  path/to/your/data  path/to/save/outputs
# ==============
in_dir=$1
out_dir=$2


# For now, let's use data in .scilpy
scil_data_download -v ERROR
in_dir=$in_dir/qball
mkdir $in_dir
cp $HOME/.scilpy/processing/dwi_crop.nii.gz $in_dir/dwi.nii.gz
cp $HOME/.scilpy/processing/1000.bval $in_dir/dwi.bval
cp $HOME/.scilpy/processing/1000.bvec $in_dir/dwi.bvec
cp $HOME/.scilpy/processing/fa_thr.nii.gz $in_dir/mask.nii.gz

# ==============
# Now let's run the tutorial
# ==============
cd $out_dir

echo "Running QBall"
scil_qball_metrics $in_dir/dwi.nii.gz $in_dir/dwi.bval $in_dir/dwi.bvec \
    --mask $in_dir/mask.nii.gz --not_all --gfa gfa.nii.gz --nufo nufo.nii.gz -v