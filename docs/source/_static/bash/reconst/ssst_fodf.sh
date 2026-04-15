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
in_dir=$in_dir/ssst
mkdir $in_dir
cp $HOME/.scilpy/processing/dwi_crop.nii.gz $in_dir/dwi.nii.gz
cp $HOME/.scilpy/processing/dwi.bval $in_dir/dwi.bval
cp $HOME/.scilpy/processing/dwi.bvec $in_dir/dwi.bvec
cp $HOME/.scilpy/processing/fa_thr.nii.gz $in_dir/mask.nii.gz
cp $HOME/.scilpy/processing/fa_thr.nii.gz $in_dir/wm_mask.nii.gz


# ==============
# Now let's run the tutorial
# ==============
cd $out_dir

echo "1 - preparing the FRF"
echo "*********************"
scil_frf_ssst $in_dir/dwi.nii.gz $in_dir/dwi.bval $in_dir/dwi.bvec frf.txt \
    --mask $in_dir/mask.nii.gz --mask_wm $in_dir/wm_mask.nii.gz -v

echo "2 - Preparing the fODF"
echo "**********************"
scil_fodf_ssst $in_dir/dwi.nii.gz $in_dir/dwi.bval $in_dir/dwi.bvec frf.txt fodf.nii.gz \
    --mask $in_dir/mask.nii.gz

echo "3 - Visualizing the fODF"
echo "************************"
# Here, the --silent flag is used to avoid opening a visualization window.
# It should be remove if you want to see the interactive visualization.
scil_viz_fodf fodf.nii.gz --silent --output fodf_ssst.png
