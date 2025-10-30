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
in_dir=$in_dir/msmt
mkdir $in_dir
cp $HOME/.scilpy/commit_amico/* $in_dir/

# Let's crop our data.
# The explanation for this section is in
# https://scilpy.readthedocs.io/en/latest/documentation/volumes_manip/cropping.html
echo "Cropping!"
echo '{' >> $out_dir/bounding_box.json
echo '    "minimums": [-20, -30, -20],'  >> $out_dir/bounding_box.json
echo '    "maximums": [20, 30, 20],' >> $out_dir/bounding_box.json
echo '    "voxel_size": [5.0, 5.0, 5.0]' >> $out_dir/bounding_box.json
echo '}' >> $out_dir/bounding_box.json
scil_volume_crop $in_dir/dwi.nii.gz $in_dir/dwi.nii.gz \
    --input_bbox $out_dir/bounding_box.json -f
scil_volume_crop $in_dir/mask.nii.gz $in_dir/mask.nii.gz \
    --input_bbox $out_dir/bounding_box.json -f
scil_volume_math convert $in_dir/mask.nii.gz $in_dir/mask.nii.gz --data_type uint8 -f

# ==============
# Now let's run the tutorial
# ==============
cd $out_dir

echo "Creating the frf"
echo "*****************"
scil_frf_msmt $in_dir/dwi.nii.gz $in_dir/dwi.bval $in_dir/dwi.bvec \
    wm_frf.txt gm_frf.txt csf_frf.txt --mask $in_dir/mask.nii.gz -v --min_nvox 1

# Currently not available in tutorial data:
# --mask_wm $in_dir/wm_mask.nii.gz --mask_gm $in_dir/gm_mask.nii.gz --mask_csf $in_dir/csf_mask.nii.gz

echo "Creating the fODF"
echo "*****************"
scil_fodf_msmt $in_dir/dwi.nii.gz $in_dir/dwi.bval $in_dir/dwi.bvec \
    wm_frf.txt gm_frf.txt csf_frf.txt --mask $in_dir/mask.nii.gz -v
