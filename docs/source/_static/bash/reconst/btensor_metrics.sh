#!/usr/bin/env bash
set -euo pipefail  # Will fail on error

# ==============
# How to run this script
#    1) Load the input data
#       https://scilpy.readthedocs.io/en/latest/documentation/getting_started.html
#    2) Call this script with
#    --->   bash btensor_metrics.sh  path/to/your/data  path/to/save/outputs
# ==============
in_dir=$1
out_dir=$2

in_dir=$in_dir/btensor

# For now, the tutorial data only contains the masks.
# Other necessary data can be obtained with:
scil_data_download -v ERROR
cp $HOME/.scilpy/btensor_testdata/* $in_dir/

# ==============
# Now let's run the tutorial
# ==============
cd $out_dir

echo "Compute metrics"
echo "*****************"
scil_btensor_metrics \
    --in_dwis $in_dir/dwi_linear.nii.gz $in_dir/dwi_planar.nii.gz $in_dir/dwi_spherical.nii.gz \
    --in_bvals $in_dir/linear.bvals $in_dir/planar.bvals $in_dir/spherical.bvals \
    --in_bvecs $in_dir/linear.bvecs $in_dir/planar.bvecs $in_dir/spherical.bvecs \
    --in_bdeltas 1 -0.5 0 --fa $in_dir/fa.nii.gz --processes 8 --mask $in_dir/mask.nii.gz
echo "Resulting files: "
ls ./