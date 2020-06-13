scil_perform_majority_vote.py
==============

::

	usage: scil_perform_majority_vote.py [-h] [--ratio_streamlines RATIO_STREAMLINES]
	                    [--ratio_voxels RATIO_VOXELS] [--same_tractogram]
	                    [--output_prefix OUTPUT_PREFIX] [--reference REFERENCE]
	                    [-f]
	                    in_bundles [in_bundles ...]
	
	Use multiple bundles to perform a voxel-wise vote (occurence across input).
	If streamlines originate from the same tractogram, streamline-wise vote
	is available.
	
	Useful to generate an average representation from bundles of a given population
	or multiple bundle segmentations (gold standard).
	
	Input tractograms must have identical header.
	
	positional arguments:
	  in_bundles            Input bundles filename.
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --ratio_streamlines RATIO_STREAMLINES
	                        Minimum vote to be considered for streamlines [0.5].
	  --ratio_voxels RATIO_VOXELS
	                        Minimum vote to be considered for voxels [0.5].
	  --same_tractogram     All bundles need to come from the same tractogram,
	                        will generate a voting for streamlines too.
	  --output_prefix OUTPUT_PREFIX
	                        Output prefix, [voting_].
	  --reference REFERENCE
	                        Reference anatomy for tck/vtk/fib/dpy file
	                        support (.nii or .nii.gz).
	  -f                    Force overwriting of the output files.
