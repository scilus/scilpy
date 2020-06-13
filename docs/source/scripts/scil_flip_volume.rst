scil_flip_volume.py
==============

::

	usage: scil_flip_volume.py [-h] [-f] input output dimension [dimension ...]
	
	Flip the volume according to the specified axis.
	
	positional arguments:
	  input       Path of the input volume (nifti).
	  output      Path of the output volume (nifti).
	  dimension   The axes you want to flip. eg: to flip the x and y axes use: x y.
	
	optional arguments:
	  -h, --help  show this help message and exit
	  -f          Force overwriting of the output files.
