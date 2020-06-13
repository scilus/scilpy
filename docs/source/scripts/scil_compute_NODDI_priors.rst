scil_compute_NODDI_priors.py
==============

::

	usage: scil_compute_NODDI_priors.py [-h] [--fa_min FA_MIN] [--fa_max FA_MAX] [--md_min MD_MIN]
	                    [--roi_radius ROI_RADIUS]
	                    [--roi_center tuple3) [tuple(3 ...]]
	                    [--output_1fiber file] [--mask_output_1fiber file]
	                    [--output_ventricles file] [--mask_output_ventricles file]
	                    [-f] [-v]
	                    in_FA in_AD in_MD
	
	Compute the axial (para_diff) and mean (iso_diff) diffusivity priors for NODDI.
	
	positional arguments:
	  in_FA                 Path to the FA volume.
	  in_AD                 Path to the axial diffusivity (AD) volume.
	  in_MD                 Path to the mean diffusivity (MD) volume.
	
	optional arguments:
	  -h, --help            show this help message and exit
	  -f                    Force overwriting of the output files.
	  -v                    If set, produces verbose output.
	
	Metrics options:
	  --fa_min FA_MIN       Minimal threshold of FA (voxels above that threshold
	                        are considered in the single fiber mask). [0.7]
	  --fa_max FA_MAX       Maximal threshold of FA (voxels under that threshold
	                        are considered in the ventricles). [0.1]
	  --md_min MD_MIN       Minimal threshold of MD in mm2/s (voxels above that
	                        threshold are considered for in the ventricles).
	                        [0.003]
	
	Regions options:
	  --roi_radius ROI_RADIUS
	                        Radius of the region used to estimate the priors. The
	                        roi will be a cube spanning from ROI_CENTER in each
	                        direction. [20]
	  --roi_center tuple(3) [tuple(3) ...]
	                        Center of the roi of size roi_radius used to estimate
	                        the priors. [center of the 3D volume]
	
	Outputs:
	  --output_1fiber file  Output path for the text file containing the single
	                        fiber average value of AD. If not set, the file will
	                        not be saved.
	  --mask_output_1fiber file
	                        Output path for single fiber mask. If not set, the
	                        mask will not be saved.
	  --output_ventricles file
	                        Output path for the text file containing the
	                        ventricles average value of MD. If not set, the file
	                        will not be saved.
	  --mask_output_ventricles file
	                        Output path for the ventricule mask. If not set, the
	                        mask will not be saved.
	
	Reference:
	    [1] Zhang H, Schneider T, Wheeler-Kingshott CA, Alexander DC.
	        NODDI: practical in vivo neurite orientation dispersion
	        and density imaging of the human brain.
	        NeuroImage. 2012 Jul 16;61:1000-16.
