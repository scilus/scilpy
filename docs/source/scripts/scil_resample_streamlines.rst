scil_resample_streamlines.py
==============

::

	usage: scil_resample_streamlines.py [-h]
	                    (--nb_pts_per_streamline NB_PTS_PER_STREAMLINE | --step_size STEP_SIZE)
	                    [--reference REFERENCE] [-f]
	                    in_tractogram out_tractogram
	
	Script to resample a set of streamlines to either a new number of points per
	streamline or to a fixed step size. WARNING: data_per_point is not carried.
	
	positional arguments:
	  in_tractogram         Streamlines input file name.
	  out_tractogram        Streamlines output file name.
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --nb_pts_per_streamline NB_PTS_PER_STREAMLINE
	                        Number of points per streamline in the output.
	  --step_size STEP_SIZE
	                        Step size in the output (in mm).
	  --reference REFERENCE
	                        Reference anatomy for tck/vtk/fib/dpy file
	                        support (.nii or .nii.gz).
	  -f                    Force overwriting of the output files.
