scil_smooth_surface.py
==============

::

	usage: scil_smooth_surface.py [-h] [-m VTS_MASK] [-n NB_STEPS] [-s STEP_SIZE] [-f] [-v]
	                    in_surface out_surface
	
	Script to smooth a surface with a Laplacian blur.
	
	step_size from 0.1 to 10 is recommended
	Smoothing_time = step_size * nb_steps
	    [1, 10] for a small smoothing
	    [10, 100] for a moderate smoothing
	    [100, 1000] for a big smoothing
	
	positional arguments:
	  in_surface            Input surface (.vtk).
	  out_surface           Output smoothed surface (.vtk).
	
	optional arguments:
	  -h, --help            show this help message and exit
	  -m VTS_MASK, --vts_mask VTS_MASK
	                        Vertices mask, where to apply the flow (.npy).
	  -n NB_STEPS, --nb_steps NB_STEPS
	                        Number of steps for laplacian smooth [2].
	  -s STEP_SIZE, --step_size STEP_SIZE
	                        Laplacian smooth step size [5.0].
	  -f                    Force overwriting of the output files.
	  -v                    If set, produces verbose output.
	
	References:
	[1] St-Onge, E., Daducci, A., Girard, G. and Descoteaux, M. 2018.
	    Surface-enhanced tractography (SET). NeuroImage.
