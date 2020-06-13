scil_set_response_function.py
==============

::

	usage: scil_set_response_function.py [-h] [--no_factor] [-f] input tuple output
	
	Replace the fiber response function in the FRF file.
	Use this script when you want to use a fixed response function
	and keep the mean b0.
	
	The FRF file is obtained from scil_compute_ssst_frf.py
	
	positional arguments:
	  input        Path of the FRF file.
	  tuple        Replace the response function with
	               this fiber response function x 10**-4 (e.g. 15,4,4).
	  output       Path of the new FRF file.
	
	optional arguments:
	  -h, --help   show this help message and exit
	  --no_factor  If supplied, the fiber response function is
	               evaluated without the x 10**-4 factor. [False].
	  -f           Force overwriting of the output files.
