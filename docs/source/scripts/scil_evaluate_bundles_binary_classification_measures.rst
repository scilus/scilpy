scil_evaluate_bundles_binary_classification_measures.py
==============

::

	usage: scil_evaluate_bundles_binary_classification_measures.py [-h]
	                    [--streamlines_measures GOLD_STANDARD_STREAMLINES TRACTOGRAM]
	                    [--voxels_measures GOLD_STANDARD_MASK TRACKING MASK]
	                    [--processes NBR] [--reference REFERENCE] [-v]
	                    [--indent INDENT] [--sort_keys] [-f]
	                    in_bundles [in_bundles ...] out_json
	
	Evaluate binary classification measures between gold standard and bundles.
	All tractograms must be in the same space (aligned to one reference)
	The measures can be applied to voxel-wise or streamline-wise representation.
	
	A gold standard must be provided for the desired representation.
	A gold standard would be a segmentation from an expert or a group of experts.
	If only the streamline-wise representation is provided without a voxel-wise
	gold standard, it will be computed from the provided streamlines.
	At least one of the two representations is required.
	
	The gold standard tractogram is the tractogram (whole brain most likely) from
	which the segmentation is performed.
	The gold standard tracking mask is the tracking mask used by the tractography
	algorighm to generate the gold standard tractogram.
	
	The computed binary classification measures are:
	sensitivity, specificity, precision, accuracy, dice, kappa, youden for both
	the streamline and voxel representation (if provided).
	
	positional arguments:
	  in_bundles            Path of the input bundles.
	  out_json              Path of the output json.
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --streamlines_measures GOLD_STANDARD_STREAMLINES TRACTOGRAM
	                        The gold standard bundle and the original tractogram.
	  --voxels_measures GOLD_STANDARD_MASK TRACKING MASK
	                        The gold standard mask and the original tracking mask.
	  --processes NBR       Number of sub-processes to start. 
	                        Default: [1]
	  --reference REFERENCE
	                        Reference anatomy for tck/vtk/fib/dpy file
	                        support (.nii or .nii.gz).
	  -v                    If set, produces verbose output.
	  -f                    Force overwriting of the output files.
	
	Json options:
	  --indent INDENT       Indent for json pretty print.
	  --sort_keys           Sort keys in output json.
