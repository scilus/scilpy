scil_detect_streamlines_loops.py
==============

::

	usage: scil_detect_streamlines_loops.py [-h] [--looping_tractogram LOOPING_TRACTOGRAM] [--qb]
	                    [--threshold THRESHOLD] [-a ANGLE] [-f]
	                    [--reference REFERENCE]
	                    in_tractogram out_tractogram
	
	This script can be used to remove loops in two types of streamline datasets:
	
	  - Whole brain: For this type, the script removes streamlines if they
	    make a loop with an angle of more than 360 degrees. It's possible to change
	    this angle with the -a option. Warning: Don't use --qb option for a
	    whole brain tractography.
	
	  - Bundle dataset: For this type, it is possible to remove loops and
	    streamlines outside of the bundle. For the sharp angle turn, use --qb
	    option.
	
	----------------------------------------------------------------------------
	Reference:
	QuickBundles based on [Garyfallidis12] Frontiers in Neuroscience, 2012.
	----------------------------------------------------------------------------
	
	positional arguments:
	  in_tractogram         Tractogram input file name.
	  out_tractogram        Output tractogram without loops.
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --looping_tractogram LOOPING_TRACTOGRAM
	                        If set, saves detected looping streamlines.
	  --qb                  If set, uses QuickBundles to detect
	                        outliers (loops, sharp angle turns).
	                        Should mainly be used with bundles. [False]
	  --threshold THRESHOLD
	                        Maximal streamline to bundle distance
	                        for a streamline to be considered as
	                        a tracking error. [8.0]
	  -a ANGLE              Maximum looping (or turning) angle of
	                        a streamline in degrees. [360]
	  -f                    Force overwriting of the output files.
	  --reference REFERENCE
	                        Reference anatomy for tck/vtk/fib/dpy file
	                        support (.nii or .nii.gz).
