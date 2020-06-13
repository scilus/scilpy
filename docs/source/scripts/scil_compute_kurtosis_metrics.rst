scil_compute_kurtosis_metrics.py
==============

::

	usage: scil_compute_kurtosis_metrics.py [-h] [--mask MASK] [--tolerance INT] [--min_k MIN_K]
	                    [--max_k MAX_K] [--smooth SMOOTH] [--not_all] [--ak file]
	                    [--mk file] [--rk file] [--msk file] [--dki_fa file]
	                    [--dki_md file] [--dki_ad file] [--dki_rd file]
	                    [--dki_residual file] [--msd file] [--force_b0_threshold]
	                    [-f]
	                    input bvals bvecs
	
	Script to compute the Diffusion Kurtosis Imaging (DKI) and Mean Signal DKI
	(MSDKI) metrics. DKI is a multi-shell diffusion model. The input DWI needs
	to be multi-shell, i.e. multi-bvalued.
	
	Since the diffusion kurtosis model involves the estimation of a large number
	of parameters and since the non-Gaussian components of the diffusion signal
	are more sensitive to artefacts, you should really denoise your DWI volume
	before using this DKI script (e.g. scil_run_nlmeans.py). Moreover, to remove
	biases due to fiber dispersion, fiber crossings and other mesoscopic properties
	of the underlying tissue, MSDKI does a powder-average of DWI for all
	directions, thus removing the orientational dependencies and creating an
	alternative mean kurtosis map.
	
	DKI is also known to be vulnerable to artefacted voxels induced by the
	low radial diffusivities of aligned white matter (CC, CST voxels). Since it is
	very hard to capture non-Gaussian information due to the low decays in radial
	direction, its kurtosis estimates have very low robustness.
	Noisy kurtosis estimates tend to be negative and its absolute values can have
	order of magnitudes higher than the typical kurtosis values. Consequently,
	these negative kurtosis values will heavily propagate to the mean and radial
	kurtosis metrics. This is well-reported in [Rafael Henriques MSc thesis 2012,
	chapter 3]. Two ways to overcome this issue: i) compute the kurtosis values
	from powder-averaged MSDKI, and ii) perform 3D Gaussian smoothing. On
	powder-averaged signal decays, you don't have this low diffusivity issue and
	your kurtosis estimates have much higher precision (additionally they are
	independent to the fODF).
	
	By default, will output all available metrics, using default names. Specific
	names can be specified using the metrics flags that are listed in the "Metrics
	files flags" section. If --not_all is set, only the metrics specified
	explicitly by the flags will be output.
	
	This script directly comes from the DIPY example gallery and references
	therein.
	[1] examples_built/reconst_dki/#example-reconst-dki
	[2] examples_built/reconst_msdki/#example-reconst-msdki
	
	positional arguments:
	  input                 Path of the input multi-shell DWI dataset.
	  bvals                 Path of the bvals file, in FSL format.
	  bvecs                 Path of the bvecs file, in FSL format.
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --mask MASK           Path to a binary mask.
	                        Only data inside the mask will be used for computations and reconstruction. 
	                        [Default: None]
	  --tolerance INT, -t INT
	                        The tolerated distance between the b-values to extract
	                        and the actual b-values [Default: 20].
	  --min_k MIN_K         Minimum kurtosis value in the output maps 
	                        (ak, mk, rk). In theory, -3/7 is the min kurtosis 
	                        limit for regions that consist of water confined 
	                        to spherical pores (see DIPY example and 
	                        documentation) [Default: 0.0].
	  --max_k MAX_K         Maximum kurtosis value in the output maps 
	                        (ak, mk, rk). In theory, 10 is the max kurtosis
	                        limit for regions that consist of water confined
	                        to spherical pores (see DIPY example and 
	                        documentation) [Default: 3.0].
	  --smooth SMOOTH       Smooth input DWI with a 3D Gaussian filter with 
	                        full-width-half-max (fwhm). Kurtosis fitting is 
	                        sensitive and outliers occur easily. According to
	                        tests on HCP, CB_Brain, Penthera3T, this smoothing
	                        is thus turned ON by default with fwhm=2.5. 
	                        [Default: 2.5].
	  --not_all             If set, will only save the metrics explicitly 
	                        specified using the other metrics flags. 
	                        [Default: not set].
	  --force_b0_threshold  If set, the script will continue even if the minimum bvalue is suspiciously high ( > 20)
	  -f                    Force overwriting of the output files.
	
	Metrics files flags:
	  --ak file             Output filename for the axial kurtosis.
	  --mk file             Output filename for the mean kurtosis.
	  --rk file             Output filename for the radial kurtosis.
	  --msk file            Output filename for the mean signal kurtosis.
	  --dki_fa file         Output filename for the fractional anisotropy from DKI.
	  --dki_md file         Output filename for the mean diffusivity from DKI.
	  --dki_ad file         Output filename for the axial diffusivity from DKI.
	  --dki_rd file         Output filename for the radial diffusivity from DKI.
	
	Quality control files flags:
	  --dki_residual file   Output filename for the map of the residual of the tensor fit.
	  --msd file            Output filename for the mean signal diffusion (powder-average).
