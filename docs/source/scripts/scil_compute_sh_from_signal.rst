scil_compute_sh_from_signal.py
==============

::

	usage: scil_compute_sh_from_signal.py [-h] [--sh_order SH_ORDER]
	                    [--sh_basis {descoteaux07,tournier07}] [--smooth SMOOTH]
	                    [--use_attenuation] [--force_b0_threshold] [--mask MASK]
	                    [-f]
	                    dwi bvals bvecs output
	
	Script to compute the SH coefficient directly on the raw DWI signal.
	
	positional arguments:
	  dwi                   Path of the dwi volume.
	  bvals                 Path of the bvals file, in FSL format.
	  bvecs                 Path of the bvecs file, in FSL format.
	  output                Name of the output SH file to save.
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --sh_order SH_ORDER   SH order to fit (int). [4]
	  --sh_basis {descoteaux07,tournier07}
	                        Spherical harmonics basis used for the SH coefficients.
	                        Must be either 'descoteaux07' or 'tournier07' [descoteaux07]:
	                            'descoteaux07': SH basis from the Descoteaux et al.
	                                              MRM 2007 paper
	                            'tournier07'  : SH basis from the Tournier et al.
	                                              NeuroImage 2007 paper.
	  --smooth SMOOTH       Lambda-regularization coefficient in the SH fit (float). [0.006]
	  --use_attenuation     If set, will use signal attenuation before fitting the SH (i.e. divide by the b0).
	  --force_b0_threshold  If set, the script will continue even if the minimum bvalue is suspiciously high ( > 20)
	  --mask MASK           Path to a binary mask.
	                        Only data inside the mask will be used for computations and reconstruction 
	  -f                    Force overwriting of the output files.
