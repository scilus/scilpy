scil_compute_fodf.py
==============

::

	usage: scil_compute_fodf.py [-h] [--sh_order int] [--mask] [--processes NBR]
	                    [--not_all] [--force_b0_threshold]
	                    [--sh_basis {descoteaux07,tournier07}] [--fodf file]
	                    [--peaks file] [--peak_indices file] [-f]
	                    input bvals bvecs frf_file
	
	Script to compute Constrained Spherical Deconvolution (CSD) fiber ODFs.
	
	By default, will output all possible files, using default names. Specific names
	can be specified using the file flags specified in the "File flags" section.
	
	If --not_all is set, only the files specified explicitly by the flags
	will be output.
	
	See [Tournier et al. NeuroImage 2007] and [Cote et al Tractometer MedIA 2013]
	for quantitative comparisons with Sharpening Deconvolution Transform (SDT)
	
	positional arguments:
	  input                 Path of the input diffusion volume.
	  bvals                 Path of the bvals file, in FSL format.
	  bvecs                 Path of the bvecs file, in FSL format.
	  frf_file              Path of the FRF file
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --sh_order int        SH order used for the CSD. (Default: 8)
	  --mask                Path to a binary mask. Only the data inside the mask will be used for computations and reconstruction.
	  --processes NBR       Number of sub processes to start. Default : cpu count
	  --not_all             If set, only saves the files specified using the file flags. (Default: False)
	  --force_b0_threshold  If set, the script will continue even if the minimum bvalue is suspiciously high ( > 20)
	  --sh_basis {descoteaux07,tournier07}
	                        Spherical harmonics basis used for the SH coefficients.
	                        Must be either 'descoteaux07' or 'tournier07' [descoteaux07]:
	                            'descoteaux07': SH basis from the Descoteaux et al.
	                                              MRM 2007 paper
	                            'tournier07'  : SH basis from the Tournier et al.
	                                              NeuroImage 2007 paper.
	  -f                    Force overwriting of the output files.
	
	File flags:
	  --fodf file           Output filename for the fiber ODF coefficients.
	  --peaks file          Output filename for the extracted peaks.
	  --peak_indices file   Output filename for the generated peaks indices on the sphere.
