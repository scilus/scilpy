scil_compute_streamlines_length_stats.py
==============

::

	usage: scil_compute_streamlines_length_stats.py [-h] [--reference REFERENCE] [--indent INDENT]
	                    [--sort_keys]
	                    in_bundle
	
	Compute streamlines min, mean and max length, as well as
	standard deviation of length in mm.
	
	positional arguments:
	  in_bundle             Fiber bundle file.
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --reference REFERENCE
	                        Reference anatomy for tck/vtk/fib/dpy file
	                        support (.nii or .nii.gz).
	
	Json options:
	  --indent INDENT       Indent for json pretty print.
	  --sort_keys           Sort keys in output json.
