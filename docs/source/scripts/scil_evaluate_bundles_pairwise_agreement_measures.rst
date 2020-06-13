scil_evaluate_bundles_pairwise_agreement_measures.py
==============

::

	usage: scil_evaluate_bundles_pairwise_agreement_measures.py [-h] [--streamline_dice] [--disable_streamline_distance]
	                    [--single_compare SINGLE_COMPARE] [--keep_tmp]
	                    [--processes NBR] [--reference REFERENCE]
	                    [--indent INDENT] [--sort_keys] [-f]
	                    in_bundles [in_bundles ...] out_json
	
	Evaluate pair-wise similarity measures of bundles.
	All tractograms must be in the same space (aligned to one reference)
	
	For the voxel representation the computed similarity measures are:
	bundle_adjacency_voxels, dice_voxels, w_dice_voxels, density_correlation
	volume_overlap, volume_overreach
	The same measures are also evluated for the endpoints.
	
	For the streamline representation the computed similarity measures are:
	bundle_adjacency_streamlines, dice_streamlines, streamlines_count_overlap,
	streamlines_count_overreach
	
	positional arguments:
	  in_bundles            Path of the input bundles.
	  out_json              Path of the output json file.
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --streamline_dice     Streamlines-wise Dice coefficient will be computed 
	                        Tractograms must be identical [False].
	  --disable_streamline_distance
	                        Will not compute the streamlines distance 
	                        [False].
	  --single_compare SINGLE_COMPARE
	                        Compare inputs to this single file.
	  --keep_tmp            Will not delete the tmp folder at the end.
	  --processes NBR       Number of sub-processes to start. 
	                        Default: [1]
	  --reference REFERENCE
	                        Reference anatomy for tck/vtk/fib/dpy file
	                        support (.nii or .nii.gz).
	  -f                    Force overwriting of the output files.
	
	Json options:
	  --indent INDENT       Indent for json pretty print.
	  --sort_keys           Sort keys in output json.
