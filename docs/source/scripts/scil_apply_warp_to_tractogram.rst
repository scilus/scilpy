scil_apply_warp_to_tractogram.py
==============

::

	usage: scil_apply_warp_to_tractogram.py [-h] [--cut_invalid | --remove_invalid | --keep_invalid]
	                    [-f] [--reference REFERENCE]
	                    in_moving_tractogram in_target_file in_deformation
	                    out_tractogram
	
	Warp tractogram using a non linear deformation from an ANTs deformation field.
	
	For more information on how to use the various registration scripts
	see the doc/tractogram_registration.md readme file
	
	Applying a deformation field to tractogram can lead to invalid streamlines (out
	of the bounding box), three strategies are available:
	1) default, crash at saving if invalid streamlines are present
	2) --keep_invalid, save invalid streamlines. Leave it to the user to run
	    scil_remove_invalid_streamlines.py if needed.
	3) --remove_invalid, automatically remove invalid streamlines before saving.
	    Should not remove more than a few streamlines.
	
	positional arguments:
	  in_moving_tractogram  Path to the tractogram to be transformed.
	  in_target_file        Path to the reference target file (trk or nii).
	  in_deformation        Path to the file containing a deformation field.
	  out_tractogram        Output filename of the transformed tractogram.
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --cut_invalid         Cut invalid streamlines rather than removing them.
	                        Keep the longest segment only.
	  --remove_invalid      Remove the streamlines landing out of the bounding box.
	  --keep_invalid        Keep the streamlines landing out of the bounding box.
	  -f                    Force overwriting of the output files.
	  --reference REFERENCE
	                        Reference anatomy for tck/vtk/fib/dpy file
	                        support (.nii or .nii.gz).
