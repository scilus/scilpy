scil_convert_gradients_mrtrix_to_fsl.py
==============

::

	usage: scil_convert_gradients_mrtrix_to_fsl.py [-h] [-f] [-v] mrtrix_enc fsl_bval fsl_bvec
	
	Script to convert bval/bvec MRtrix style to FSL style.
	
	positional arguments:
	  mrtrix_enc  Path to the gradient directions encoding file. (.b)
	  fsl_bval    Path to output FSL b-value file (.bval).
	  fsl_bvec    Path to output FSL gradient directions file (.bvec).
	
	optional arguments:
	  -h, --help  show this help message and exit
	  -f          Force overwriting of the output files.
	  -v          If set, produces verbose output.
