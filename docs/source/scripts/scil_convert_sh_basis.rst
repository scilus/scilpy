scil_convert_sh_basis.py
==============

::

	usage: scil_convert_sh_basis.py [-h] [-f] input_sh output_name {descoteaux07,tournier07}
	
	    Convert a SH file between the two commonly used bases
	    ('descoteaux07' or 'tournier07'). The specified basis corresponds to the
	    input data basis.
	
	positional arguments:
	  input_sh              Input SH filename. (nii or nii.gz)
	  output_name           Name of the output file.
	  {descoteaux07,tournier07}
	                        Spherical harmonics basis used for the SH coefficients.
	                        Must be either 'descoteaux07' or 'tournier07' [descoteaux07]:
	                            'descoteaux07': SH basis from the Descoteaux et al.
	                                              MRM 2007 paper
	                            'tournier07'  : SH basis from the Tournier et al.
	                                              NeuroImage 2007 paper.
	
	optional arguments:
	  -h, --help            show this help message and exit
	  -f                    Force overwriting of the output files.
