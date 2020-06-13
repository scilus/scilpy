scil_compute_streamlines_density_map.py
==============

::

	usage: scil_compute_streamlines_density_map.py [-h] [--binary [FIXED_VALUE]] [--reference REFERENCE] [-f]
	                    in_bundle out_img
	
	Compute a density map from a streamlines file.
	
	A specific value can be assigned instead of using the tract count.
	
	This script correctly handles compressed streamlines.
	
	positional arguments:
	  in_bundle             Tractogram filename. Format must be one of 
	                        trk, tck, vtk, fib, dpy.
	  out_img               path of the output image file.
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --binary [FIXED_VALUE]
	                        If set, will store the same value for all intersected voxels, creating a binary map.
	                        When set without a value, 1 is used.
	                        If a value is given, will be used as the stored value.
	  --reference REFERENCE
	                        Reference anatomy for tck/vtk/fib/dpy file
	                        support (.nii or .nii.gz).
	  -f                    Force overwriting of the output files.
