scil_compute_pft.py
==============

::

	usage: scil_compute_pft.py [-h] [--algo {det,prob}] [--step STEP_SIZE]
	                    [--min_length MIN_LENGTH] [--max_length MAX_LENGTH]
	                    [--theta THETA] [--act] [--sfthres SF_THRESHOLD]
	                    [--sfthres_init SF_THRESHOLD_INIT]
	                    [--sh_basis {descoteaux07,tournier07}]
	                    [--npv NPV | --nt NT] [--particles PARTICLES]
	                    [--back BACK_TRACKING] [--forward FORWARD_TRACKING]
	                    [--compress COMPRESS] [--all] [--seed SEED] [-f]
	                    [--save_seeds] [-v]
	                    sh_file seed_file map_include_file map_exclude_file
	                    output_file
	
	Local streamline HARDI tractography including Particle Filtering tracking.
	
	The tracking is done inside partial volume estimation maps and uses the
	particle filtering tractography (PFT) algorithm. See
	scil_compute_maps_for_particle_filter_tracking.py
	to generate PFT required maps.
	
	Streamlines longer than min_length and shorter than max_length are kept.
	The tracking direction is chosen in the aperture cone defined by the
	previous tracking direction and the angular constraint.
	
	Algo 'det': the maxima of the spherical function (SF) the most closely aligned
	to the previous direction.
	Algo 'prob': a direction drawn from the empirical distribution function defined
	from the SF.
	
	Default parameters as suggested in [1].
	
	positional arguments:
	  sh_file               Spherical harmonic file. Data must be aligned with 
	                        seed_file (isotropic resolution, nifti, see --basis).
	  seed_file             Seeding mask (isotropic resolution, nifti).
	  map_include_file      The probability map of ending the streamline and 
	                        including it in the output (CMC, PFT [1]). 
	                        (isotropic resolution, nifti).
	  map_exclude_file      The probability map of ending the streamline and 
	                        excluding it in the output (CMC, PFT [1]). 
	                        (isotropic resolution, nifti).
	  output_file           Streamline output file (must be trk or tck).
	
	Generic options:
	  -h, --help            show this help message and exit
	
	Tracking options:
	  --algo {det,prob}     Algorithm to use (must be "det" or "prob"). [prob]
	  --step STEP_SIZE      Step size in mm. [0.5]
	  --min_length MIN_LENGTH
	                        Minimum length of a streamline in mm. [10.0]
	  --max_length MAX_LENGTH
	                        Maximum length of a streamline in mm. [300.0]
	  --theta THETA         Maximum angle between 2 steps. ["det"=45, "prob"=20]
	  --act                 If set, uses anatomically-constrained tractography (ACT)
	                        instead of continuous map criterion (CMC).
	  --sfthres SF_THRESHOLD
	                        Spherical function relative threshold. [0.1]
	  --sfthres_init SF_THRESHOLD_INIT
	                        Spherical function relative threshold value for the 
	                        initial direction. [0.5]
	  --sh_basis {descoteaux07,tournier07}
	                        Spherical harmonics basis used for the SH coefficients.
	                        Must be either 'descoteaux07' or 'tournier07' [descoteaux07]:
	                            'descoteaux07': SH basis from the Descoteaux et al.
	                                              MRM 2007 paper
	                            'tournier07'  : SH basis from the Tournier et al.
	                                              NeuroImage 2007 paper.
	
	Seeding options:
	  When no option is provided, uses --npv 1.
	
	  --npv NPV             Number of seeds per voxel.
	  --nt NT               Total number of seeds to use.
	
	PFT options:
	  --particles PARTICLES
	                        Number of particles to use for PFT. [15]
	  --back BACK_TRACKING  Length of PFT back tracking in mm. [2.0]
	  --forward FORWARD_TRACKING
	                        Length of PFT forward tracking in mm. [1.0]
	
	Output options:
	  --compress COMPRESS   If set, will compress streamlines. The parameter
	                        value is the distance threshold. A rule of thumb
	                        is to set it to 0.1mm for deterministic
	                        streamlines and 0.2mm for probabilitic streamlines.
	  --all                 If set, keeps "excluded" streamlines.
	                        NOT RECOMMENDED, except for debugging.
	  --seed SEED           Random number generator seed.
	  -f                    Force overwriting of the output files.
	  --save_seeds          If set, save the seeds used for the tracking in the data_per_streamline property of the tractogram.
	
	Logging options:
	  -v                    If set, produces verbose output.
	
	References: [1] Girard, G., Whittingstall K., Deriche, R., and Descoteaux, M. (2014). Towards quantitative connectivity analysis: reducing tractography biases. Neuroimage, 98, 266-278.
