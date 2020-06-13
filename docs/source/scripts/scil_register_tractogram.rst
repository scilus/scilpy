scil_register_tractogram.py
==============

::

	usage: scil_register_tractogram.py [-h] [--out_name OUT_NAME] [--only_rigid]
	                    [--moving_tractogram_ref MOVING_TRACTOGRAM_REF]
	                    [--static_tractogram_ref STATIC_TRACTOGRAM_REF] [-f] [-v]
	                    moving_tractogram static_tractogram
	
	Generate a linear transformation matrix from the registration of
	2 tractograms. Typically, this script is run before
	scil_apply_transform_to_tractogram.py.
	
	For more informations on how to use the various registration scripts
	see the doc/tractogram_registration.md readme file
	
	positional arguments:
	  moving_tractogram     Path of the moving tractogram.
	  static_tractogram     Path of the target tractogram.
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --out_name OUT_NAME   Filename of the transformation matrix, 
	                        the registration type will be appended as a suffix,
	                        [<out_name>_<affine/rigid>.txt]
	  --only_rigid          Will only use a rigid transformation, uses affine by default.
	  --moving_tractogram_ref MOVING_TRACTOGRAM_REF
	                        Reference anatomy for moving_tractogram (if tck/vtk/fib/dpy) file
	                        support (.nii or .nii.gz).
	  --static_tractogram_ref STATIC_TRACTOGRAM_REF
	                        Reference anatomy for static_tractogram (if tck/vtk/fib/dpy) file
	                        support (.nii or .nii.gz).
	  -f                    Force overwriting of the output files.
	  -v                    If set, produces verbose output.
	
	References:
	[1] E. Garyfallidis, O. Ocegueda, D. Wassermann, M. Descoteaux
	Robust and efficient linear registration of white-matter fascicles in the
	space of streamlines, NeuroImage, Volume 117, 15 August 2015, Pages 124-140
	(http://www.sciencedirect.com/science/article/pii/S1053811915003961)
