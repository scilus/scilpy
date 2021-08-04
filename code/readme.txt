Direct multiresolution tracking
------------------------------------------------------------------------------------------------------------------------------

All in scilpy

######################
scil_compute_tracking_multiresolution.py 
######################

Local streamline HARDI tractography script. 

With --mask_multiresolution and --voxel_size, tracking can be done in lower resolution for regions specified in mask.

######################
In tracking package

local_tracking.py
######################

Methods now have resampled_tracker and region_mr as arguments.

The tracking in lower resolution gets done in _get_line_binary.



Multiresolution tracking with anatomical priors
--------------------------------------------------------------------------------------------------------------------------------

All written in bash

######################
compute_full_tractogram_multiresolution.sh
######################

Compute gaussian pyramid with anatomical priors with full tractogram at lower resolution

* Must have Antoine's modifications for generate_priors_from_bundles in order to have adequate priors (changing of tractogram reference)

######################
compute_quickbundle_multiresolution.sh
######################

Compute gaussian pyramid with anatomical priors for each bundle at lower resolution

* Must have Antoine's modifications for generate_priors_from_bundles in order to have adequate priors (changing of tractogram reference)

######################
compare_bundles_connectivity.sh
######################

Compute graph with tc_bundle and fc_bundle from tractogram scores

######################
compare_tractogram_dice_overlap.sh
######################

Compute graph with tractogram overlap and dice from each bundle from tractogram scores

######################
compute_tract_score.sh
######################

Compute tracking and scoring of all fodf provided 



Asymmetric filtering
-------------------------------------------------------------------------------------------------------------------------------

The code used is from Charles Poirier asym-filtering-mask branch 
scil_execute_asymmetric_filtering.py with --out_sym





