#!/usr/bin/env bash
set -euo pipefail  # Will fail on error

# ==============
# How to run this script
#    1) Load the input data
#       https://scilpy.readthedocs.io/en/latest/documentation/getting_started.html
#    2) Call this script with
#    --->   bash aodf_scripts.sh  path/to/your/data  path/to/save/outputs
# ==============
in_folder=$1
out_folder=$2

# For now, let's use data in .scilpy
scil_data_download
mkdir $in_folder/aodf_data
in_folder=$in_folder/aodf_data
cp $HOME/.scilpy/processing/fa_thr.nii.gz $in_folder/brainmask.nii.gz
cp $HOME/.scilpy/processing/fodf_descoteaux07.nii.gz $in_folder/fodf.nii.gz


# ==============
# Now let's run the tutorial
# Running only the first option
# ==============
cd $out_folder

echo "Creating the aodf"
echo "*****************"
scil_sh_to_aodf $in_folder/fodf.nii.gz afodf.nii.gz -v

echo "Computing metrics"
echo "*****************"
scil_aodf_metrics afodf.nii.gz --mask $in_folder/brainmask.nii.gz -v