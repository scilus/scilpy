scil_run_nlmeans.py
==============

::

	usage: scil_run_nlmeans.py [-h] [--mask] [--sigma float] [--log LOGFILE]
	                    [--processes int] [-v] [-f]
	                    input output number_coils
	
	Script to denoise a dataset with the Non Local Means algorithm.
	
	positional arguments:
	  input            Path of the image file to denoise.
	  output           Path to save the denoised image file.
	  number_coils     Number of receiver coils of the scanner.
	                   Use N=1 in the case of a SENSE (GE, Philips) reconstruction and 
	                   N >= 1 for GRAPPA reconstruction (Siemens). N=4 works well for the 1.5T
	                   in Sherbrooke. Use N=0 if the noise is considered Gaussian distributed.
	
	optional arguments:
	  -h, --help       show this help message and exit
	  --mask           Path to a binary mask. Only the data inside the mask will be used for computations
	  --sigma float    The standard deviation of the noise to use instead of computing  it automatically.
	  --log LOGFILE    If supplied, name of the text file to store the logs.
	  --processes int  Number of sub processes to start. Default: Use all cores.
	  -v, --verbose    Print more info. Default : Print only warnings.
	  -f               Force overwriting of the output files.
