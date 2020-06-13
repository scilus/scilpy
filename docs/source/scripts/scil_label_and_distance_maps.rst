scil_label_and_distance_maps.py
==============

::

	usage: scil_label_and_distance_maps.py [-h] [-f] [--reference REFERENCE]
	                    in_bundle in_centroid output_label output_distance
	
	Compute assignment map from bundle and centroid streamline.
	This script can be very memory hungry on large fiber bundle.
	
	positional arguments:
	  in_bundle             Fiber bundle file
	  in_centroid           Centroid streamline associated to input fiber bundle
	  output_label          Output (.npz) file containing the label of the nearest point on the centroid streamline for each point of the bundle
	  output_distance       Output (.npz) file containing the distance (in mm) to the nearest centroid streamline for each point of the bundle
	
	optional arguments:
	  -h, --help            show this help message and exit
	  -f                    Force overwriting of the output files.
	  --reference REFERENCE
	                        Reference anatomy for tck/vtk/fib/dpy file
	                        support (.nii or .nii.gz).
