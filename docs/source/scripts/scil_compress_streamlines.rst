scil_compress_streamlines.py
==============

::

	usage: scil_compress_streamlines.py [-h] [-e ERROR_RATE] [-f] in_tractogram out_tractogram
	
	Compress tractogram by removing collinear (or almost) points.
	
	The compression threshold represents the maximum distance (in mm) to the
	original position of the point.
	
	positional arguments:
	  in_tractogram   Path of the input tractogram file (trk or tck).
	  out_tractogram  Path of the output tractogram file (trk or tck).
	
	optional arguments:
	  -h, --help      show this help message and exit
	  -e ERROR_RATE   Maximum compression distance in mm [0.1].
	  -f              Force overwriting of the output files.
