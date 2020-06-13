scil_flip_gradients.py
==============

::

	usage: scil_flip_gradients.py [-h] (--fsl | --mrtrix) [-f]
	                    gradient_sampling_file flipped_sampling_file dimension
	                    [dimension ...]
	
	Flip one or more axes of the gradient sampling matrix. It will be saved in
	the same format as input gradient sampling file.
	
	positional arguments:
	  gradient_sampling_file
	                        Path to gradient sampling file. (.bvec or .b)
	  flipped_sampling_file
	                        Path to the flipped gradient sampling file.
	  dimension             The axes you want to flip. eg: to flip the x and y axes use: x y.
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --fsl                 Specify fsl format.
	  --mrtrix              Specify mrtrix format.
	  -f                    Force overwriting of the output files.
