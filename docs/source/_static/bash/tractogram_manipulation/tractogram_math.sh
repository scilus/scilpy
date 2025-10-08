#!/usr/bin/env bash
set -euo pipefail  # Will fail on error

# --------------------
# Logical operation
# --------------------

# Merging two tractograms (using all streamlines from both)
# If a streamline is exactly the same in both tractograms, it will be kept once.
scil_tractogram_count_streamlines $tractogram1
scil_tractogram_count_streamlines $tractogram2
scil_tractogram_math union $tractogram1 $tractogram2 union.trk
scil_tractogram_count_streamlines union.trk

# Finding streamlines belonging in two tractograms (intersection)
scil_tractogram_math intersection $tractogram1 $tractogram2 intersection.trk

# Finding streamlines that are in $tractogram1 but not in $tractogram2
scil_tractogram_math difference $tractogram1 $tractogram2 difference1-2.trk



# --------------------
# Resampling
# --------------------

# Downsampling: keeping only 20 streamlines
scil_tractogram_resample $tractogram1 20 tractogram_downsampled.trk

# Downsampling per cluster
scil_tractogram_resample $tractogram1 20 tractogram_downsampled.trk --downsample_per_cluster

# Upsampling: uses randomly picked streamlines to create new similar ones
# by moving a little bit each point.
# We will also give a --tube_radius option to make sure final streamlines
# do not deviate too much from original trajectory.
scil_tractogram_resample $tractogram1 1000 --point_wise_std

# Splitting into 3 chunks
scil_tractogram_split --nb_chunks 3 $tractogram1 out_split_3_part

# Splitting into chunks of 100 streamlines
scil_tractogram_split --chunk_size 100 $tractogram1 out_split_100_part

# Splitting per cluster
scil_tractogram_split --nb_chunk 2 --split_per_cluster $tractogram1 out_split_2QB_part

# --------------------
# Flipping
# --------------------
scil_tractogram_flip $tractogram1 flipped_x.trk x
