scil_apply_bias_field_on_dwi.py
==============

::

	usage: scil_apply_bias_field_on_dwi.py [-h] [--mask MASK] [-f] in_dwi in_bias_field out_name
	
	Apply bias field correction to DWI. This script doesn't compute the bias
	field itself. It ONLY applies an existing bias field. Use the ANTs
	N4BiasFieldCorrection executable to compute the bias field
	
	positional arguments:
	  in_dwi         DWI Nifti image.
	  in_bias_field  Bias field Nifti image.
	  out_name       Corrected DWI Nifti image.
	
	optional arguments:
	  -h, --help     show this help message and exit
	  --mask MASK    Apply bias field correction only in the region defined by the mask.
	  -f             Force overwriting of the output files.
