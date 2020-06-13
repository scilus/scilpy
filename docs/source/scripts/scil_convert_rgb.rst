scil_convert_rgb.py
==============

::

	usage: scil_convert_rgb.py [-h] [-f] in_image out_image
	
	Converts a RGB image encoded as a 4D image to a RGB image encoded as
	a 3D image, or vice versa.
	
	Typically, most software tools used in the SCIL (including MI-Brain) use
	the former, while Trackvis uses the latter.
	
	Input
	-Case 1: 4D image where the 4th dimension contains 3 values.
	-Case 2: 3D image, in Trackvis format where each voxel contains a
	         tuple of 3 elements, one for each value.
	
	Output
	-Case 1: 3D image, in Trackvis format where each voxel contains a
	         tuple of 3 elements, one for each value (uint8).
	-Case 2: 4D image where the 4th dimension contains 3 values (uint8).
	
	positional arguments:
	  in_image    name of input RGB image.
	              Either 4D or 3D image.
	  out_image   name of output RGB image.
	              Either 3D or 4D image.
	
	optional arguments:
	  -h, --help  show this help message and exit
	  -f          Force overwriting of the output files.
