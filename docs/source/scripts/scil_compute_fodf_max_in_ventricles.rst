scil_compute_fodf_max_in_ventricles.py
==============

::

	usage: scil_compute_fodf_max_in_ventricles.py [-h] [--fa_t FA_THRESHOLD] [--md_t MD_THRESHOLD]
	                    [--max_value_output file] [--mask_output file]
	                    [--sh_basis {descoteaux07,tournier07}] [-v] [-f]
	                    fODFs FA MD
	
	Script to compute the maximum fODF in the ventricles. The ventricules are
	estimated from a MD and FA threshold.
	
	This allows to clip the noise of fODF using an absolute thresold.
	
	positional arguments:
	  fODFs                 Path of the fODF volume in spherical harmonics (SH).
	  FA                    Path to the FA volume.
	  MD                    Path to the mean diffusivity (MD) volume.
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --fa_t FA_THRESHOLD   Maximal threshold of FA (voxels under that threshold are considered for evaluation, [0.1]).
	  --md_t MD_THRESHOLD   Minimal threshold of MD in mm2/s (voxels above that threshold are considered for evaluation, [0.003]).
	  --max_value_output file
	                        Output path for the text file containing the value. If not set the file will not be saved.
	  --mask_output file    Output path for the ventricule mask. If not set, the mask will not be saved.
	  --sh_basis {descoteaux07,tournier07}
	                        Spherical harmonics basis used for the SH coefficients.
	                        Must be either 'descoteaux07' or 'tournier07' [descoteaux07]:
	                            'descoteaux07': SH basis from the Descoteaux et al.
	                                              MRM 2007 paper
	                            'tournier07'  : SH basis from the Tournier et al.
	                                              NeuroImage 2007 paper.
	  -v                    If set, produces verbose output.
	  -f                    Force overwriting of the output files.
	
	[1] Dell'Acqua, Flavio, et al. "Can spherical deconvolution provide more
	    information than fiber orientations? Hindrance modulated orientational
	    anisotropy, a true‐tract specific index to characterize white matter
	    diffusion." Human brain mapping 34.10 (2013): 2464-2483.
