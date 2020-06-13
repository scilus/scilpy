scil_compute_NODDI.py
==============

::

	usage: scil_compute_NODDI.py [-h] [--in_mask IN_MASK] [--out_dir OUT_DIR]
	                    [--b_thr B_THR] [--para_diff PARA_DIFF]
	                    [--iso_diff ISO_DIFF] [--lambda1 LAMBDA1]
	                    [--lambda2 LAMBDA2]
	                    [--save_kernels DIRECTORY | --load_kernels DIRECTORY]
	                    [--processes NBR] [-f] [-v]
	                    in_dwi in_bval in_bvec
	
	Compute NODDI [1] maps using AMICO.
	Multi-shell DWI necessary.
	
	positional arguments:
	  in_dwi                DWI file acquired with a NODDI compatible protocol
	                        (single-shell data not suited).
	  in_bval               b-values filename, in FSL format (.bval).
	  in_bvec               b-vectors filename, in FSL format (.bvec).
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --in_mask IN_MASK     Brain mask filename.
	  --out_dir OUT_DIR     Output directory for the NODDI results. [results]
	  --b_thr B_THR         Limit value to consider that a b-value is on an
	                        existing shell. Above this limit, the b-value is
	                        placed on a new shell. This includes b0s values.
	  --processes NBR       Number of sub-processes to start. Default: [1]
	  -f                    Force overwriting of the output files.
	  -v                    If set, produces verbose output.
	
	Model options:
	  --para_diff PARA_DIFF
	                        Axial diffusivity (AD) in the CC. [0.0017]
	  --iso_diff ISO_DIFF   Mean diffusivity (MD) in ventricles. [0.003]
	  --lambda1 LAMBDA1     First regularization parameter. [2]
	  --lambda2 LAMBDA2     Second regularization parameter. [0.001]
	
	Kernels options:
	  --save_kernels DIRECTORY
	                        Output directory for the COMMIT kernels.
	  --load_kernels DIRECTORY
	                        Input directory where the COMMIT kernels are located.
	
	Reference:
	    [1] Zhang H, Schneider T, Wheeler-Kingshott CA, Alexander DC.
	        NODDI: practical in vivo neurite orientation dispersion
	        and density imaging of the human brain.
	        NeuroImage. 2012 Jul 16;61:1000-16.
