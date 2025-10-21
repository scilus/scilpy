#!/usr/bin/env bash
set -euo pipefail  # Will fail on error

# ==============
# How to run this script
#    1) Load the input data
#       https://scilpy.readthedocs.io/en/latest/documentation/getting_started.html
#    2) Call this script with
#    --->   bash btensor_scripts.sh  $in_dir/your/data  $in_dir/save/outputs
# ==============
in_dir=$1
out_folder=$2


# For now, let's use data in .scilpy
scil_data_download -v ERROR
in_dir=$in_dir/mti
mkdir $in_dir
cp $HOME/.scilpy/ihMT/B1* $in_dir/
cp $HOME/.scilpy/ihMT/echo-1* $in_dir/
cp $HOME/.scilpy/ihMT/mask_resample.nii.gz $in_dir/mask.nii.gz

# ==============
# Now let's run the tutorial
# ==============
cd $out_folder


# 2. Basic usage
echo "Basic usage: ihmt"
echo "******************"
scil_mti_maps_ihMT $out_folder \
        --in_altnp $in_dir/*altnp*.nii.gz \
        --in_altpn $in_dir/*altpn*.nii.gz \
        --in_negative $in_dir/*neg*.nii.gz \
        --in_positive $in_dir/echo*pos*.nii.gz \
        --in_mtoff_pd $in_dir/echo*mtoff*.nii.gz \
        --in_mtoff_t1 $in_dir/echo*T1w*.nii.gz \
        --mask $in_dir/mask.nii.gz \
        --in_jsons $in_dir/echo*mtoff*.json $in_dir/echo*T1w*.json

# Or:
echo "Or for MT only"
echo "**************"
scil_mti_maps_MT $out_folder \
    --in_positive $in_dir/echo*pos*.nii.gz \
    --in_negative $in_dir/echo*neg*.nii.gz \
    --in_mtoff_pd $in_dir/echo*mtoff*.nii.gz \
    --in_mtoff_t1 $in_dir/echo*T1w*.nii.gz \
    --mask $in_dir/mask.nii.gz \
    --in_jsons $in_dir/echo*mtoff*.json $in_dir/echo*T1w*.json
