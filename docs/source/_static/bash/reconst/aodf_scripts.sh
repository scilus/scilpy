#!/usr/bin/env bash
set -euo pipefail  # Will fail on error

# ==============
# How to run this script
#    1) Load the input data
#       https://scilpy.readthedocs.io/en/latest/documentation/getting_started.html
#    2) Call this script with
#    --->   bash aodf_scripts.sh  path/to/your/data  path/to/save/outputs
# ==============
in_dir=$1
out_dir=$2

# For now, let's use data in .scilpy
scil_data_download -v ERROR
mkdir $in_dir/aodf_data
in_dir=$in_dir/aodf_data
cp $HOME/.scilpy/processing/fa_thr.nii.gz $in_dir/brainmask.nii.gz
cp $HOME/.scilpy/processing/fodf_descoteaux07.nii.gz $in_dir/fodf.nii.gz

# Let's crop our data.
# The explanation for this section is in
# https://scilpy.readthedocs.io/en/latest/documentation/volumes_manip/cropping.html
echo "Cropping!"
echo '{' >> $out_dir/bounding_box.json
echo '    "minimums": [-20, -30, -20],'  >> $out_dir/bounding_box.json
echo '    "maximums": [20, 30, 20],' >> $out_dir/bounding_box.json
echo '    "voxel_size": [2.5, 2.5, 2.5]' >> $out_dir/bounding_box.json
echo '}' >> $out_dir/bounding_box.json
scil_volume_crop $in_dir/fodf.nii.gz $in_dir/fodf.nii.gz \
    --input_bbox $out_dir/bounding_box.json -f
scil_volume_crop $in_dir/brainmask.nii.gz $in_dir/brainmask.nii.gz \
    --input_bbox $out_dir/bounding_box.json -f
scil_volume_math convert $in_dir/brainmask.nii.gz $in_dir/brainmask.nii.gz \
  --data_type uint8 -f

# ==============
# Now let's run the tutorial
# Running only the first option
# ==============
cd $out_dir

echo "Creating the aodf"
echo "*****************"
scil_sh_to_aodf $in_dir/fodf.nii.gz afodf.nii.gz -v --sphere repulsion100

echo "Computing metrics"
echo "*****************"
scil_aodf_metrics afodf.nii.gz --mask $in_dir/brainmask.nii.gz -v
