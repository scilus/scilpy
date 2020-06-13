scil_compute_local_tracking.py
==============

::

	usage: scil_compute_local_tracking.py [-h] [--algo {det,prob}] [--step STEP_SIZE]
	                    [--min_length MIN_LENGTH] [--max_length MAX_LENGTH]
	                    [--theta THETA] [--sfthres SF_THRESHOLD]
	                    [--sh_basis {descoteaux07,tournier07}]
	                    [--npv NPV | --nt NT]
	                    [--sphere {repulsion100,repulsion200,repulsion724,symmetric362,symmetric642,symmetric724}]
	                    [--compress COMPRESS] [--seed SEED] [-f] [--save_seeds]
	                    [-v]
	                    sh_file seed_file mask_file output_file
	
	Local streamline HARDI tractography.
	The tracking direction is chosen in the aperture cone defined by the
	previous tracking direction and the angular constraint.
	
	Algo 'eudx': the peak from the spherical function (SF) most closely aligned
	to the previous direction.
	Algo 'det': the maxima of the spherical function (SF) the most closely aligned
	to the previous direction.
	Algo 'prob': a direction drawn from the empirical distribution function defined
	from the SF.
	
	positional arguments:
	  sh_file               Spherical harmonic file. 
	                        (isotropic resolution, nifti, see --basis).
	  seed_file             Seeding mask (isotropic resolution, nifti).
	  mask_file             Seeding mask(isotropic resolution, nifti).
	                        Tracking will stop outside this mask.
	  output_file           Streamline output file (must be trk or tck).
	
	Generic options:
	  -h, --help            show this help message and exit
	  --sphere {repulsion100,repulsion200,repulsion724,symmetric362,symmetric642,symmetric724}
	                        Set of directions to be used for tracking.
	
	Tracking options:
	  --algo {det,prob}     Algorithm to use (must be "det" or "prob"). [prob]
	  --step STEP_SIZE      Step size in mm. [0.5]
	  --min_length MIN_LENGTH
	                        Minimum length of a streamline in mm. [10.0]
	  --max_length MAX_LENGTH
	                        Maximum length of a streamline in mm. [300.0]
	  --theta THETA         Maximum angle between 2 steps. ["eudx"=60, det"=45, "prob"=20]
	  --sfthres SF_THRESHOLD
	                        Spherical function relative threshold. [0.1]
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
	
	Output options:
	  --compress COMPRESS   If set, will compress streamlines. The parameter
	                        value is the distance threshold. A rule of thumb
	                        is to set it to 0.1mm for deterministic
	                        streamlines and 0.2mm for probabilitic streamlines.
	  --seed SEED           Random number generator seed.
	  -f                    Force overwriting of the output files.
	  --save_seeds          If set, save the seeds used for the tracking in the data_per_streamline property of the tractogram.
	
	Logging options:
	  -v                    If set, produces verbose output.
