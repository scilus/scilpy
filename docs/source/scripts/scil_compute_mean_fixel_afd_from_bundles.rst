scil_compute_mean_fixel_afd_from_bundles.py
==============

::

	usage: scil_compute_mean_fixel_afd_from_bundles.py [-h] [--length_weighting] [--reference REFERENCE]
	                    [--sh_basis {descoteaux07,tournier07}] [-f]
	                    in_bundle in_fODF afd_mean_map rd_mean_map
	
	Compute the mean Apparent Fiber Density (AFD) and mean Radial fODF (radfODF)
	maps along a bundle.
	
	This is the "real" fixel-based fODF amplitude along every streamline
	of the bundle provided, averaged at every voxel.
	Radial fODF comes for free from the mathematics, it is the great circle
	integral of the fODF orthogonal to the fixel of interest, similar to
	a Funk-Radon transform. Hence, radfODF is the fixel-based or HARDI-based
	generalization of the DTI radial diffusivity and AFD
	the generalization of axial diffusivity.
	
	Please use a bundle file rather than a whole tractogram.
	
	positional arguments:
	  in_bundle             Path of the bundle file.
	  in_fODF               Path of the fODF volume in spherical harmonics (SH).
	  afd_mean_map          Path of the output mean AFD map.
	  rd_mean_map           Path of the output mean radfODF map.
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --length_weighting    if set, will weigh the AFD values according to segment lengths. [False]
	  --reference REFERENCE
	                        Reference anatomy for tck/vtk/fib/dpy file
	                        support (.nii or .nii.gz).
	  --sh_basis {descoteaux07,tournier07}
	                        Spherical harmonics basis used for the SH coefficients.
	                        Must be either 'descoteaux07' or 'tournier07' [descoteaux07]:
	                            'descoteaux07': SH basis from the Descoteaux et al.
	                                              MRM 2007 paper
	                            'tournier07'  : SH basis from the Tournier et al.
	                                              NeuroImage 2007 paper.
	  -f                    Force overwriting of the output files.
	
	Reference:
	    [1] Raffelt, D., Tournier, JD., Rose, S., Ridgway, GR., Henderson, R.,
	        Crozier, S., Salvado, O., & Connelly, A. (2012).
	        Apparent Fibre Density: a novel measure for the analysis of
	        diffusion-weighted magnetic resonance images. NeuroImage, 59(4), 3976--3994.
