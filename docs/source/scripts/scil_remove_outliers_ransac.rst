scil_remove_outliers_ransac.py
==============

::

	usage: scil_remove_outliers_ransac.py [-h] [--min_fit MIN_FIT] [--max_iter MAX_ITER]
	                    [--fit_thr FIT_THR] [-v] [-f]
	                    in_image out_image
	
	Remove outliers from image using the RANSAC algorithm.
	The RANSAC algorithm parameters are sensitive to the input data.
	
	NOTE: Current default parameters are tuned for ad/md/rd images only.
	
	positional arguments:
	  in_image             Nifti image.
	  out_image            Corrected Nifti image.
	
	optional arguments:
	  -h, --help           show this help message and exit
	  --min_fit MIN_FIT    The minimum number of data values required to fit the model. [50]
	  --max_iter MAX_ITER  The maximum number of iterations allowed in the algorithm. [1000]
	  --fit_thr FIT_THR    Threshold value for determining when a data point fits a model. [0.01]
	  -v                   If set, produces verbose output.
	  -f                   Force overwriting of the output files.
