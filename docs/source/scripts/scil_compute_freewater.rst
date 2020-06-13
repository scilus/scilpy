scil_compute_freewater.py
==============

::

	usage: scil_compute_freewater.py [-h] [--in_mask IN_MASK] [--out_dir OUT_DIR]
	                    [--b_thr B_THR] [--para_diff PARA_DIFF]
	                    [--iso_diff ISO_DIFF] [--perp_diff_min PERP_DIFF_MIN]
	                    [--perp_diff_max PERP_DIFF_MAX] [--lambda1 LAMBDA1]
	                    [--lambda2 LAMBDA2]
	                    [--save_kernels DIRECTORY | --load_kernels DIRECTORY]
	                    [--mouse] [--processes NBR] [-f] [-v]
	                    in_dwi in_bval in_bvec
	
	Compute Free Water maps [1] using AMICO.
	This script supports both single and multi-shell data.
	
	positional arguments:
	  in_dwi                DWI file.
	  in_bval               b-values filename, in FSL format (.bval).
	  in_bvec               b-vectors filename, in FSL format (.bvec).
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --in_mask IN_MASK     Brain mask filename.
	  --out_dir OUT_DIR     Output directory for the Free Water results.
	                        [current_directory]
	  --b_thr B_THR         Limit value to consider that a b-value is on an
	                        existing shell. Above this limit, the b-value is
	                        placed on a new shell. This includes b0s values.
	  --mouse               If set, use mouse fitting profile.
	  --processes NBR       Number of sub-processes to start. Default: [1]
	  -f                    Force overwriting of the output files.
	  -v                    If set, produces verbose output.
	
	Model options:
	  --para_diff PARA_DIFF
	                        Axial diffusivity (AD) in the CC. [0.0015]
	  --iso_diff ISO_DIFF   Mean diffusivity (MD) in ventricles. [0.003]
	  --perp_diff_min PERP_DIFF_MIN
	                        Radial diffusivity (RD) minimum. [0.0001]
	  --perp_diff_max PERP_DIFF_MAX
	                        Radial diffusivity (RD) maximum. [0.0007]
	  --lambda1 LAMBDA1     First regularization parameter. [0.0]
	  --lambda2 LAMBDA2     Second regularization parameter. [0.001]
	
	Kernels options:
	  --save_kernels DIRECTORY
	                        Output directory for the COMMIT kernels.
	  --load_kernels DIRECTORY
	                        Input directory where the COMMIT kernels are located.
	
	Reference:
	    [1] Pasternak 0, Sochen N, Gur Y, Intrator N, Assaf Y.
	        Free water elimination and mapping from diffusion mri.
	        Magn Reson Med. 62 (3) (2009) 717-730.
