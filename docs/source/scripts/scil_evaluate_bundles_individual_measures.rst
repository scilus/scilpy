scil_evaluate_bundles_individual_measures.py
==============

::

	usage: scil_evaluate_bundles_individual_measures.py [-h] [--reference REFERENCE] [--processes NBR]
	                    [--indent INDENT] [--sort_keys] [-f]
	                    in_bundles [in_bundles ...] out_json
	
	Evaluate basic measurements of bundles, all at once.
	All tractograms must be in the same space (aligned to one reference)
	
	The computed measures are:
	volume, volume_endpoints, streamlines_count, avg_length, std_length,
	min_length, max_length, mean_curvature
	
	positional arguments:
	  in_bundles            Path of the input bundles.
	  out_json              Path of the output file.
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --reference REFERENCE
	                        Reference anatomy for tck/vtk/fib/dpy file
	                        support (.nii or .nii.gz).
	  --processes NBR       Number of sub-processes to start. 
	                        Default: [1]
	  -f                    Force overwriting of the output files.
	
	Json options:
	  --indent INDENT       Indent for json pretty print.
	  --sort_keys           Sort keys in output json.
