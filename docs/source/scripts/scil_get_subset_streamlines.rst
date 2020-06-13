scil_get_subset_streamlines.py
==============

::

	usage: scil_get_subset_streamlines.py [-h] [--seed SEED] [--reference REFERENCE] [-f]
	                    in_tractogram max_num_streamlines out_tractogram
	
	Script to get a subset of n streamlines from a tractogram.
	
	positional arguments:
	  in_tractogram         Streamlines input file name.
	  max_num_streamlines   Maximum number of streamlines to output.
	  out_tractogram        Streamlines output file name.
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --seed SEED           Use a specific random seed for the resampling.
	  --reference REFERENCE
	                        Reference anatomy for tck/vtk/fib/dpy file
	                        support (.nii or .nii.gz).
	  -f                    Force overwriting of the output files.
