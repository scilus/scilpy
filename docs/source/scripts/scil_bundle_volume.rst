scil_bundle_volume.py
==============

::

	usage: scil_bundle_volume.py [-h] [--reference REFERENCE] [--indent INDENT]
	                    [--sort_keys]
	                    in_bundle
	
	Compute bundle volume in mm³. This script supports anisotropic voxels
	resolution. Volume is estimated by counting the number of voxels occupied by
	the bundle and multiplying it by the volume of a single voxel.
	
	This estimation is typically performed at resolution around 1mm³.
	
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
