#!/usr/bin/env bash
set -euo pipefail  # Will fail on error

# --------------------
# Resampling / compressing
# --------------------

# Choosing the number of points
scil_tractogram_resample_nb_points $tractogram1 20 resampled_20pts.trk

# Compressing
scil_tractogram_compress


# --------------------
# Cutting streamlines
# --------------------

# Outside a mask
scil_tractogram_cut_streamlines $tractogram1 cut_streamlines.trk \
    --mask $mask --min_length 20

# Outside labels
ROI1=3
ROI2=6
scil_tractogram_cut_streamlines $tractogram1 cut_streamlines.trk \
    --labels $my_parcellation --label_ids $ROI1 $ROI2
