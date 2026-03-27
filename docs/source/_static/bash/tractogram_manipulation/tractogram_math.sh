#!/usr/bin/env bash
set -euo pipefail  # Will fail on error

# ==============
# How to run this script
#    1) Load the input data
#       https://scilpy.readthedocs.io/en/latest/documentation/getting_started.html
#    2) Call this script with
#    --->   bash tractogram_math.sh  path/to/your/data  path/to/save/outputs
# ==============
in_dir=$1
out_dir=$2


# Let's use two different tractograms available in the data:
tractogram1=$in_dir/sub-01/sub-01__cst_L_part1.trk
tractogram2=$in_dir/sub-01/sub-01__cst_L_part2.trk

# --------------------
# A. Logical operation
# --------------------
cd $out_dir

echo "Counting streamlines in tractogram1 and tractogram2"
scil_tractogram_count_streamlines $tractogram1
scil_tractogram_count_streamlines $tractogram2

echo "Merging two tractograms (using all streamlines from both)"
scil_tractogram_math union $tractogram1 $tractogram2 union.trk
scil_tractogram_count_streamlines union.trk

echo "Finding streamlines belonging in two tractograms (intersection)"
scil_tractogram_math intersection $tractogram1 $tractogram2 intersection.trk

echo "Finding streamlines that are in tractogram1 but not in tractogram2"
scil_tractogram_math difference $tractogram1 $tractogram2 difference1-2.trk



# --------------------
# B. Resampling
# --------------------

echo "Downsampling"
scil_tractogram_resample $tractogram1 200 tractogram_downsampled.trk
scil_tractogram_resample $tractogram1 200 tractogram_downsampled2.trk \
    --downsample_per_cluster -v --qbx_thresholds 3

echo "Upsampling"
scil_tractogram_resample $tractogram1 4000 tractogram_upsampled.trk \
    --point_wise_std 5 -v --tube_radius 4

echo "Splitting"
# scil_tractogram_split --nb_chunks 3 $tractogram1 out_split_3_part
scil_tractogram_split --chunk_size 100 $tractogram1 out_split_100_part -v
scil_tractogram_split --nb_chunk 3 --split_per_cluster \
      $tractogram1 out_split_3QB_part -v --qbx_thresholds 6

# --------------------
# Flipping
# --------------------
echo "Flipping"
scil_tractogram_flip $tractogram1 flipped_x.trk x -v
