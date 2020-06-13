scil_extract_b0.py
==============

::

	usage: scil_extract_b0.py [-h] [--b0_thr B0_THR] [--all | --mean] [-v]
	                    dwi bvals bvecs output
	
	Extract B0s from DWI.
	
	The default behavior is to save the first b0 of the series.
	
	positional arguments:
	  dwi              DWI Nifti image
	  bvals            B-values file in FSL format
	  bvecs            B-vectors file in FSL format
	  output           Output b0 file(s)
	
	optional arguments:
	  -h, --help       show this help message and exit
	  --b0_thr B0_THR  All b-values with values less than or equal to b0_thr are considered as b0s i.e. without diffusion weighting
	  --all            Extract all b0. Index number will be appended to the output file
	  --mean           Extract mean b0
	  -v               If set, produces verbose output.
