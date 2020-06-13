scil_compute_centroid.py
==============

::

	usage: scil_compute_centroid.py [-h] [--nb_points NB_POINTS] [--reference REFERENCE] [-f]
	                    in_bundle out_centroid
	
	Compute a single bundle centroid, using an 'infinite' QuickBundles threshold.
	
	positional arguments:
	  in_bundle             Fiber bundle file.
	  out_centroid          Output centroid streamline filename.
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --nb_points NB_POINTS
	                        Number of points defining the centroid streamline[20].
	  --reference REFERENCE
	                        Reference anatomy for tck/vtk/fib/dpy file
	                        support (.nii or .nii.gz).
	  -f                    Force overwriting of the output files.
