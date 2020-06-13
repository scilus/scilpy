scil_compute_todi.py
==============

::

	usage: scil_compute_todi.py [-h] [--reference REFERENCE] [--sphere SPHERE]
	                    [--mask MASK] [--out_mask OUT_MASK]
	                    [--out_lw_tdi OUT_LW_TDI] [--out_lw_todi OUT_LW_TODI]
	                    [--out_lw_todi_sh OUT_LW_TODI_SH] [--sh_order SH_ORDER]
	                    [--sh_normed] [--smooth]
	                    [--sh_basis {descoteaux07,tournier07}] [-f]
	                    tract_filename
	
	Compute a length-weighted Track Orientation Density Image (TODI).
	This script can afterwards output a length-weighted Track Density Image
	(TDI) or a length-weighted TODI, based on streamlines' segments.
	
	positional arguments:
	  tract_filename        Input streamlines file.
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --reference REFERENCE
	                        Reference anatomy for tck/vtk/fib/dpy file
	                        support (.nii or .nii.gz).
	  --sphere SPHERE       sphere used for the angular discretization.
	  --mask MASK           Use the given mask
	  --out_mask OUT_MASK   Mask showing where TDI > 0.
	  --out_lw_tdi OUT_LW_TDI
	                        Output length-weighted TDI map.
	  --out_lw_todi OUT_LW_TODI
	                        Output length-weighted TODI map.
	  --out_lw_todi_sh OUT_LW_TODI_SH
	                        Output length-weighted TODI map, with SH coefficient.
	  --sh_order SH_ORDER   Order of the original SH.
	  --sh_normed           Normalize sh.
	  --smooth              Smooth todi (angular and spatial).
	  --sh_basis {descoteaux07,tournier07}
	                        Spherical harmonics basis used for the SH coefficients.
	                        Must be either 'descoteaux07' or 'tournier07' [descoteaux07]:
	                            'descoteaux07': SH basis from the Descoteaux et al.
	                                              MRM 2007 paper
	                            'tournier07'  : SH basis from the Tournier et al.
	                                              NeuroImage 2007 paper.
	  -f                    Force overwriting of the output files.
	
	    References:
	        [1] Dhollander T, Emsell L, Van Hecke W, Maes F, Sunaert S, Suetens P.
	            Track orientation density imaging (TODI) and
	            track orientation distribution (TOD) based tractography.
	            NeuroImage. 2014 Jul 1;94:312-36.
	    
