scil_flip_streamlines.py
==============

::

	usage: scil_flip_streamlines.py [-h] [--reference REFERENCE] [-f]
	                    in_tractogram out_tractogram {x,y,z} [{x,y,z} ...]
	
	Flip streamlines locally around specific axes.
	
	IMPORTANT: this script should only be used in case of absolute necessity.
	It's better to fix the real tools than to force flipping streamlines to
	have them fit in the tools.
	
	positional arguments:
	  in_tractogram         Path of the input tractogram file.
	  out_tractogram        Path of the output tractogram file.
	  {x,y,z}               The axes you want to flip. eg: to flip the x and y axes use: x y.
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --reference REFERENCE
	                        Reference anatomy for tck/vtk/fib/dpy file
	                        support (.nii or .nii.gz).
	  -f                    Force overwriting of the output files.
