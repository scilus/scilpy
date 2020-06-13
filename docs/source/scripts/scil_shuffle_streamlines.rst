scil_shuffle_streamlines.py
==============

::

	usage: scil_shuffle_streamlines.py [-h] [--seed SEED] [--reference REFERENCE] [-f]
	                    in_tractogram out_tractogram
	
	Shuffle the ordering of streamlines.
	
	positional arguments:
	  in_tractogram         Input tractography file.
	  out_tractogram        Output tractography file.
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --seed SEED           Random number generator seed [None].
	  --reference REFERENCE
	                        Reference anatomy for tck/vtk/fib/dpy file
	                        support (.nii or .nii.gz).
	  -f                    Force overwriting of the output files.
