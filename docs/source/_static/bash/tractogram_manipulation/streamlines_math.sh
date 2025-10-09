#!/usr/bin/env bash
set -euo pipefail  # Will fail on error

<<<<<<< HEAD
=======
# ==============
# How to run this script
#    1) Load the input data
#       https://scilpy.readthedocs.io/en/latest/documentation/getting_started.html
#    2) Call this script with
#    --->   bash streamlines_math.sh  path/to/your/data
#    Note that outputs will be saved in the current directory.
# ==============
in_folder=$1

# Let's use a tractogram available in the data:
tractogram1=$in_folder/sub-01/sub-01__cst_L_part1.trk
mask=$in_folder/sub-01/sub-01__small_mask_wm.nii.gz
labels=$in_folder/sub-01/sub-01__wmparc.nii.gz

>>>>>>> b3af6517 (Finishing tutorial for tractogram_math)
# --------------------
# Resampling / compressing
# --------------------
cd $out_folder

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

    # ---------------> FAILS WITH HEADERS NOT COMPATIBLE?? TODO

echo "Cutting streamlines Outside labels"
scil_labels_split_volume_by_ids $labels --out_dir labels/
ROI1=labels/2024.nii.gz
ROI2=labels/16.nii.gz
scil_tractogram_cut_streamlines $tractogram1 cut_streamlines.trk \
    --labels $labels --label_ids $ROI1 $ROI2
