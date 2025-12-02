#!/usr/bin/env bash
set -euo pipefail  # Will fail on error

# ==============
# How to run this script
#    1) Load the input data
#       https://scilpy.readthedocs.io/en/latest/documentation/getting_started.html
#    2) Call this script with
#    --->   bash streamlines_math.sh  path/to/your/data  path/to/save/outputs
# ==============
in_dir=$1
out_dir=$2

# Let's use a tractogram available in the data:
# tractogram1=$in_dir/sub-01/sub-01__cst_L_part2.trk not compatible...
tractogram1=$in_dir/sub-01/sub-01__cst_L.trk
labels=$in_dir/sub-01/sub-01__wmparc.nii.gz
mask=$in_dir/sub-01/sub-01__small_mask_wm.nii.gz

# Current wmparc is in float. Should be in int. Let's convert.
scil_volume_math convert $in_dir/sub-01/sub-01__wmparc.nii.gz \
    --data_type int16  -f $in_dir/sub-01/sub-01__wmparc.nii.gz

# --------------------
# Resampling / compressing
# --------------------
cd $out_dir

echo "Choosing the number of points"
scil_tractogram_resample_nb_points $tractogram1 resampled_20pts.trk \
    --nb_pts_per_streamline 20 -v

echo "Compressing"
scil_tractogram_compress $tractogram1 compressed.trk -v


# --------------------
# Cutting streamlines
# --------------------

echo "Cutting streamlines outside a mask"
scil_tractogram_cut_streamlines $tractogram1 cut_streamlines.trk \
    --mask $mask --min_length 20

echo "Cutting streamlines outside labels"
scil_tractogram_cut_streamlines $tractogram1 cut_streamlines2.trk \
    --labels $labels --label_ids 16 2024
