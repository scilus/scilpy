scil_generate_priors_from_bundle.py
==============

::

	usage: scil_generate_priors_from_bundle.py [-h] [--sh_basis {descoteaux07,tournier07}]
	                    [--todi_sigma {0,1,2,3,4}] [--sf_threshold SF_THRESHOLD]
	                    [--output_prefix OUTPUT_PREFIX] [--output_dir OUTPUT_DIR]
	                    [-f]
	                    bundle_filename fod_filename mask_filename
	
	Generation of priors and enhanced-FOD from an example/template bundle.
	The bundle must have been cleaned thorougly before use. The E-FOD can then
	be used for bundle-specific tractography, but not for FOD metrics.
	
	positional arguments:
	  bundle_filename       Input bundle filename.
	  fod_filename          Input FOD filename.
	  mask_filename         Mask to constrain the TODI spatial smoothing,
	                        for example a WM mask.
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --sh_basis {descoteaux07,tournier07}
	                        Spherical harmonics basis used for the SH coefficients.
	                        Must be either 'descoteaux07' or 'tournier07' [descoteaux07]:
	                            'descoteaux07': SH basis from the Descoteaux et al.
	                                              MRM 2007 paper
	                            'tournier07'  : SH basis from the Tournier et al.
	                                              NeuroImage 2007 paper.
	  --todi_sigma {0,1,2,3,4}
	                        Smooth the orientation histogram.
	  --sf_threshold SF_THRESHOLD
	                        Relative threshold for sf masking (0.0-1.0).
	  --output_prefix OUTPUT_PREFIX
	                        Add a prefix to all output filename, 
	                        default is no prefix.
	  --output_dir OUTPUT_DIR
	                        Output directory for all generated files,
	                        default is current directory.
	  -f                    Force overwriting of the output files.
	
	    References:
	        [1] Rheault, Francois, et al. "Bundle-specific tractography with
	        incorporated anatomical and orientational priors."
	        NeuroImage 186 (2019): 382-398
	    
