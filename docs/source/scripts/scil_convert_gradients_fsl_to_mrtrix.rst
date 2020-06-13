scil_convert_gradients_fsl_to_mrtrix.py
==============

::

	usage: scil_convert_gradients_fsl_to_mrtrix.py [-h] [-f] [-v] fsl_bval fsl_bvec mrtrix_enc
	
	Script to convert bval/bvec FSL style to MRtrix style.
	
	positional arguments:
	  fsl_bval    Path to FSL b-value file (.bval).
	  fsl_bvec    Path to FSL gradient directions file (.bvec).
	  mrtrix_enc  Path to gradient directions encoding file (.b).
	
	optional arguments:
	  -h, --help  show this help message and exit
	  -f          Force overwriting of the output files.
	  -v          If set, produces verbose output.
