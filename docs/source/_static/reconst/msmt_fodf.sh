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
?

# ==============
# Now let's run the tutorial
# ==============
cd $out_folder

echo "Creating the frf"
echo "*****************"
scil_frf_msmt dwi.nii.gz dwi.bval dwi.bvec wm_frf.txt gm_frf.txt \
        csf_frf.txt --mask $in_dir/brainmask.nii.gz --mask_wm $in_dir/wm_mask.nii.gz \
        --mask_gm $in_dir/gm_mask.nii.gz --mask_csf $in_dir/csf_mask.nii.gz -v

echo "Creating the fODF"
echo "*****************"
scil_fodf_msmt dwi.nii.gz dwi.bval dwi.bvec wm_fodf.txt gm_fodf.txt \
    csf_fodf.txt --mask $in_dir/brainmask.nii.gz -v
