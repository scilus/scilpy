scil_reshape_to_reference.py
==============

::

	usage: scil_reshape_to_reference.py [-h] [--interpolation {linear,nearest}] [-f]
	                    in_file ref_file out_file
	
	Reshape / reslice / resample *.nii or *.nii.gz using a reference.
	For more information on how to use the various registration scripts
	see the doc/tractogram_registration.md readme file.
	
	>>> scil_reshape_to_reference.py wmparc.mgz t1.nii.gz wmparc_t1.nii.gz \
	    --interpolation nearest
	
	positional arguments:
	  in_file               Path of the volume file to be reshaped.
	  ref_file              Path of the reference volume.
	  out_file              Output filename of the reshaped data.
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --interpolation {linear,nearest}
	                        Interpolation: "linear" or "nearest". [linear]
	  -f                    Force overwriting of the output files.
