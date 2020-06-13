scil_resample_volume.py
==============

::

	usage: scil_resample_volume.py [-h] (--ref REF | --resolution RESOLUTION | --iso_min)
	                    [--interp {nn,lin,quad,cubic}] [--enforce_dimensions] [-v]
	                    [-f]
	                    in_image out_image
	
	Script to resample a dataset to match the resolution of another
	reference dataset or to the resolution specified as in argument.
	
	positional arguments:
	  in_image              Path of the input volume.
	  out_image             Path of the resampled volume.
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --ref REF             Reference volume to resample to.
	  --resolution RESOLUTION
	                        Resolution to resample to. If the value it is set to is Y, it will resample to an isotropic resolution of Y x Y x Y.
	  --iso_min             Resample the volume to R x R x R with R being the smallest current voxel dimension 
	  --interp {nn,lin,quad,cubic}
	                        Interpolation mode.
	                        nn: nearest neighbour
	                        lin: linear
	                        quad: quadratic
	                        cubic: cubic
	                        Defaults to linear
	  --enforce_dimensions  Enforce the reference volume dimension.
	  -v                    If set, produces verbose output.
	  -f                    Force overwriting of the output files.
