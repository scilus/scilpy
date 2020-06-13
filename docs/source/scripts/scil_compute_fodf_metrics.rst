scil_compute_fodf_metrics.py
==============

::

	usage: scil_compute_fodf_metrics.py [-h] [--sphere string] [--mask] [--rt R_THRESHOLD]
	                    [--sh_basis {descoteaux07,tournier07}] [-f] [--vis]
	                    [--not_all] [--afd file] [--afd_total file]
	                    [--afd_sum file] [--nufo file] [--peaks file]
	                    fODFs a_threshold
	
	Script to compute the maximum Apparent Fiber Density (AFD), the fiber ODFs
	orientations (peaks) and the Number of Fiber Orientations (NuFO) maps from
	fiber ODFs.
	
	AFD_max map is the maximal fODF amplitude for each voxel.
	
	NuFO is the the number of maxima of the fODF with an ABSOLUTE amplitude above
	the threshold set using --at, AND an amplitude above the RELATIVE threshold
	set using --rt.
	
	The --at argument should be set to a value which is 1.5 times the maximal
	value of the fODF in the ventricules. This can be obtained with the
	compute_fodf_max_in_ventricules.py script.
	
	By default, will output all possible files, using default names. Specific names
	can be specified using the file flags specified in the "File flags" section.
	
	If --not_all is set, only the files specified explicitly by the flags will be
	output.
	
	See [Raffelt et al. NeuroImage 2012] and [Dell'Acqua et al HBM 2013] for the
	definitions.
	
	positional arguments:
	  fODFs                 Path of the fODF volume in spherical harmonics (SH).
	  a_threshold           WARNING!!! EXTREMELY IMPORTANT PARAMETER, VARIABLE ACROSS DATASETS!!!
	                        Absolute threshold on fODF amplitude.
	                        This value should set to approximately 1.5 to 2 times the maximum
	                        fODF amplitude in isotropic voxels (ex. ventricles).
	                        compute_fodf_max_in_ventricles.py can be used to find the maximal value.
	                        See [Dell'Acqua et al HBM 2013].
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --sphere string       Discrete sphere to use in the processing. [repulsion724].
	  --mask                Path to a binary mask. Only the data inside the mask will be used for computations and reconstruction [None].
	  --rt R_THRESHOLD      Relative threshold on fODF amplitude in percentage  [0.1].
	  --sh_basis {descoteaux07,tournier07}
	                        Spherical harmonics basis used for the SH coefficients.
	                        Must be either 'descoteaux07' or 'tournier07' [descoteaux07]:
	                            'descoteaux07': SH basis from the Descoteaux et al.
	                                              MRM 2007 paper
	                            'tournier07'  : SH basis from the Tournier et al.
	                                              NeuroImage 2007 paper.
	  -f                    Force overwriting of the output files.
	  --vis                 Export map for better visualization in FiberNavigator.
	                        !WARNING! these maps should not be used to compute statistics  [False].
	  --not_all             If set, only saves the files specified using the file flags  [False].
	
	File flags:
	  --afd file            Output filename for the AFD_max map.
	  --afd_total file      Output filename for the AFD_total map (SH coeff = 0).
	  --afd_sum file        Output filename for the sum of all peak contributions (sum of fODF lobes on the sphere).
	  --nufo file           Output filename for the NuFO map.
	  --peaks file          Output filename for the extracted peaks.
