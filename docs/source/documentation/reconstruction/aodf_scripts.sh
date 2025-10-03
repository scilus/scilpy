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


# ==============
# Now let's run the tutorial
# ==============
cd $out_folder

echo "Creating the aodf"
echo "*****************"

# Option 1. Default options.
scil_sh_to_aodf $in_folder/fodf.nii.gz afodf.nii.gz -v

# Option 2. GPU. If you have access to GPU, instead, prefer these options:
# scil_sh_to_aodf $in_folder/fodf.nii.gz afodf.nii.gz --use_opencl --device gpu -v

echo "Computing metrics"
echo "*****************"

scil_aodf_metrics afodf.nii.gz --mask $in_folder/brainmask.nii.gz -v