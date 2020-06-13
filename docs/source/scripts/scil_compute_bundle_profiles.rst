scil_compute_bundle_profiles.py
==============

::

	usage: scil_compute_bundle_profiles.py [-h]
	                    [--in_centroid IN_CENTROID | --nb_pts_per_streamline NB_PTS_PER_STREAMLINE]
	                    [--indent INDENT] [--sort_keys] [--reference REFERENCE]
	                    [-f]
	                    in_bundle in_metrics [in_metrics ...]
	
	Compute bundle profiles and their statistics along streamlines.
	
	positional arguments:
	  in_bundle             Fiber bundle file to compute the bundle profiles on.
	  in_metrics            Metric(s) on which to compute the bundle profiles.
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --in_centroid IN_CENTROID
	                        If provided it will be used to make sure all streamlines go in the same direction. 
	                        Also, number of points per streamline will be set according to centroid.
	  --nb_pts_per_streamline NB_PTS_PER_STREAMLINE
	                        If centroid not provided, resample each streamline to this number of points [20].
	  --reference REFERENCE
	                        Reference anatomy for tck/vtk/fib/dpy file
	                        support (.nii or .nii.gz).
	  -f                    Force overwriting of the output files.
	
	Json options:
	  --indent INDENT       Indent for json pretty print.
	  --sort_keys           Sort keys in output json.
