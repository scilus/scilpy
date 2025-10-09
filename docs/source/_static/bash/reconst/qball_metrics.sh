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
out_folder=$2


# For now, let's use data in .scilpy
scil_data_download
cp -r $HOME/.scilpy/btensor_testdata $in_dir/
in_dir=$in_dir/btensor_testdata/
#       -----------------> I DON'T HAVE MASKS. FAILS.

# ==============
# Now let's run the tutorial
# ==============
cd $out_folder

scil_qball_metrics $in_dir/dwi.nii.gz $in_dir/dwi.bval $in_dir/dwi.bvec \
    --mask $in_dir/brainmask.nii.gz --not_all --gfa gfa.nii.gz --nufo nufo.nii.gz