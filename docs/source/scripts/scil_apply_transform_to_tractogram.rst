scil_apply_transform_to_tractogram.py
==============

::

	usage: scil_apply_transform_to_tractogram.py [-h] [--inverse]
	                    [--cut_invalid | --remove_invalid | --keep_invalid]
	                    [--reference REFERENCE] [-f]
	                    in_moving_tractogram in_target_file in_transfo
	                    out_tractogram
	
	Transform tractogram using an affine/rigid transformation.
	
	For more information on how to use the various registration scripts
	see the doc/tractogram_registration.md readme file
	
	Applying transformation to tractogram can lead to invalid streamlines (out of
	the bounding box), three strategies are available:
	1) default, crash at saving if invalid streamlines are present
	2) --keep_invalid, save invalid streamlines. Leave it to the user to run
	    scil_remove_invalid_streamlines.py if needed.
	3) --remove_invalid, automatically remove invalid streamlines before saving.
	    Should not remove more than a few streamlines.
	
	positional arguments:
	  in_moving_tractogram  Path of the tractogram to be transformed.
	  in_target_file        Path of the reference target file (trk or nii).
	  in_transfo            Path of the file containing the 4x4 
	                        transformation, matrix (.txt, .npy or .mat).
	                        See the script description for more information.
	  out_tractogram        Output tractogram filename (transformed data).
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --inverse             Apply the inverse transformation.
	  --cut_invalid         Cut invalid streamlines rather than removing them.
	                        Keep the longest segment only.
	  --remove_invalid      Remove the streamlines landing out of the bounding box.
	  --keep_invalid        Keep the streamlines landing out of the bounding box.
	  --reference REFERENCE
	                        Reference anatomy for tck/vtk/fib/dpy file
	                        support (.nii or .nii.gz).
	  -f                    Force overwriting of the output files.
