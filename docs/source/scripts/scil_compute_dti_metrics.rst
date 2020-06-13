scil_compute_dti_metrics.py
==============

::

	usage: scil_compute_dti_metrics.py [-h] [-f] [--mask MASK] [--method method_name] [--not_all]
	                    [--ad file] [--evecs file] [--evals file] [--fa file]
	                    [--ga file] [--md file] [--mode file] [--norm file]
	                    [--rgb file] [--rd file] [--tensor file]
	                    [--non-physical file] [--pulsation string]
	                    [--residual file] [--force_b0_threshold]
	                    input bvals bvecs
	
	Script to compute all of the Diffusion Tensor Imaging (DTI) metrics.
	
	By default, will output all available metrics, using default names. Specific
	names can be specified using the metrics flags that are listed in the "Metrics
	files flags" section.
	
	If --not_all is set, only the metrics specified explicitly by the flags
	will be output. The available metrics are:
	
	fractional anisotropy (FA), geodesic anisotropy (GA), axial diffusivisty (AD),
	radial diffusivity (RD), mean diffusivity (MD), mode, red-green-blue colored
	FA (rgb), principal tensor e-vector and tensor coefficients (dxx, dxy, dxz,
	dyy, dyz, dzz).
	
	For all the quality control metrics such as residual, physically implausible
	signals, pulsation and misalignment artifacts, see
	[J-D Tournier, S. Mori, A. Leemans. Diffusion Tensor Imaging and Beyond.
	MRM 2011].
	
	positional arguments:
	  input                 Path of the input diffusion volume.
	  bvals                 Path of the bvals file, in FSL format.
	  bvecs                 Path of the bvecs file, in FSL format.
	
	optional arguments:
	  -h, --help            show this help message and exit
	  -f                    Force overwriting of the output files.
	  --mask MASK           Path to a binary mask.
	                        Only data inside the mask will be used for computations and reconstruction. (Default: None)
	  --method method_name  Tensor fit method.
	                        WLS for weighted least squares
	                        LS for ordinary least squares
	                        NLLS for non-linear least-squares
	                        restore for RESTORE robust tensor fitting. (Default: WLS)
	  --not_all             If set, will only save the metrics explicitly specified using the other metrics flags. (Default: not set).
	  --force_b0_threshold  If set, the script will continue even if the minimum bvalue is suspiciously high ( > 20)
	
	Metrics files flags:
	  --ad file             Output filename for the axial diffusivity.
	  --evecs file          Output filename for the eigenvectors of the tensor.
	  --evals file          Output filename for the eigenvalues of the tensor.
	  --fa file             Output filename for the fractional anisotropy.
	  --ga file             Output filename for the geodesic anisotropy.
	  --md file             Output filename for the mean diffusivity.
	  --mode file           Output filename for the mode.
	  --norm file           Output filename for the tensor norm.
	  --rgb file            Output filename for the colored fractional anisotropy.
	  --rd file             Output filename for the radial diffusivity.
	  --tensor file         Output filename for the tensor coefficients.
	
	Quality control files flags:
	  --non-physical file   Output filename for the voxels with physically implausible signals 
	                        where the mean of b=0 images is below one or more diffusion-weighted images.
	  --pulsation string    Standard deviation map across all diffusion-weighted images and across b=0 images if more than one is available.
	                        Shows pulsation and misalignment artifacts.
	  --residual file       Output filename for the map of the residual of the tensor fit.
