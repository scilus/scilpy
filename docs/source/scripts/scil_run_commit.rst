scil_run_commit.py
==============

::

	usage: scil_run_commit.py [-h] [--b_thr B_THR] [--nbr_dir NBR_DIR]
	                    [--nbr_iter NBR_ITER] [--in_peaks IN_PEAKS]
	                    [--in_tracking_mask IN_TRACKING_MASK] [--ball_stick]
	                    [--para_diff PARA_DIFF]
	                    [--perp_diff PERP_DIFF [PERP_DIFF ...]]
	                    [--iso_diff ISO_DIFF [ISO_DIFF ...]]
	                    [--keep_whole_tractogram] [--threshold_weights THRESHOLD]
	                    [--save_kernels DIRECTORY | --load_kernels DIRECTORY]
	                    [--processes NBR] [-f] [-v]
	                    in_tractogram in_dwi in_bval in_bvec out_dir
	
	Convex Optimization Modeling for Microstructure Informed Tractography (COMMIT)
	estimates, globally, how a given tractogram explains the DWI in terms of signal
	fit, assuming a certain forward microstructure model. It assigns a weight to
	each streamline, which represents how well it explains the DWI signal globally.
	The default forward microstructure model is stick-zeppelin-ball, which requires
	multi-shell data and a peak file (principal fiber directions in each voxel,
	typically from a field of fODFs).
	
	It is possible to use the ball-and-stick model for single-shell data. In this
	case, the peak file is not mandatory.
	
	The output from COMMIT is:
	- fit_NRMSE.nii.gz
	    fiting error (Normalized Root Mean Square Error)
	- fit_RMSE.nii.gz
	    fiting error (Root Mean Square Error)
	- results.pickle
	    Dictionary containing the experiment parameters and final weights
	- compartment_EC.nii.gz (Extra-Cellular)
	- compartment_IC.nii.gz (Intra-Cellular)
	- compartment_ISO.nii.gz (isotropic volume fraction (freewater comportment))
	    Each of COMMIT compartments
	- commit_weights.txt
	    Text file containing the commit weights for each streamline of the
	    input tractogram.
	- essential.trk / non_essential.trk
	    Tractograms containing the streamlines below or equal (essential) and
	    above (non_essential) the --threshold_weights argument.
	
	This script can divide the input tractogram in two using a threshold to apply
	on the streamlines' weight. Typically, the threshold should be 0, keeping only
	streamlines that have non-zero weight and that contribute to explain the DWI
	signal. Streamlines with 0 weight are essentially not necessary according to
	COMMIT.
	
	positional arguments:
	  in_tractogram         Input tractogram (.trk or .tck or .h5).
	  in_dwi                Diffusion-weighted images used by COMMIT (.nii.gz).
	  in_bval               b-values in the FSL format (.bval).
	  in_bvec               b-vectors in the FSL format (.bvec).
	  out_dir               Output directory for the COMMIT maps.
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --b_thr B_THR         Limit value to consider that a b-value is on an existing shell. Above this limit, the b-value is placed on a new shell. This includes b0s values.
	  --nbr_dir NBR_DIR     Number of directions, on the half of the sphere,
	                        representing the possible orientations of the response functions [500].
	  --nbr_iter NBR_ITER   Maximum number of iterations [500].
	  --in_peaks IN_PEAKS   Peaks file representing principal direction(s) locally,
	                         typically coming from fODFs. This file is mandatory for the default
	                         stick-zeppelin-ball model, when used with multi-shell data.
	  --in_tracking_mask IN_TRACKING_MASK
	                        Binary mask where tratography was allowed.
	                        If not set, uses a binary mask computed from the streamlines.
	  --processes NBR       Number of sub-processes to start. 
	                        Default: [1]
	  -f                    Force overwriting of the output files.
	  -v                    If set, produces verbose output.
	
	Model options:
	  --ball_stick          Use the ball&Stick model.
	                        Disable the zeppelin compartment for single-shell data.
	  --para_diff PARA_DIFF
	                        Parallel diffusivity in mm^2/s.
	                        Default for ball_stick: 1.7E-3
	                        Default for stick_zeppelin_ball: 1.7E-3
	  --perp_diff PERP_DIFF [PERP_DIFF ...]
	                        Perpendicular diffusivity in mm^2/s.
	                        Default for ball_stick: None
	                        Default for stick_zeppelin_ball: [1.19E-3, 0.85E-3, 0.51E-3, 0.17E-3]
	  --iso_diff ISO_DIFF [ISO_DIFF ...]
	                        Istropic diffusivity in mm^2/s.
	                        Default for ball_stick: [2.0E-3]
	                        Default for stick_zeppelin_ball: [1.7E-3, 3.0E-3]
	
	Tractogram options:
	  --keep_whole_tractogram
	                        Save a tractogram copy with streamlines weights in the data_per_streamline
	                        [default: False].
	  --threshold_weights THRESHOLD
	                        Split the tractogram in two; essential and
	                        nonessential, based on the provided threshold [0.0].
	                         Use None to skip this step.
	
	Kernels options:
	  --save_kernels DIRECTORY
	                        Output directory for the COMMIT kernels.
	  --load_kernels DIRECTORY
	                        Input directory where the COMMIT kernels are located.
