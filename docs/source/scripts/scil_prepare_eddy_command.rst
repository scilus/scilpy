scil_prepare_eddy_command.py
==============

::

	usage: scil_prepare_eddy_command.py [-h] [--topup TOPUP] [--eddy_cmd {eddy_openmp,eddy_cuda}]
	                    [--b0_thr B0_THR] [--encoding_direction {x,y,z}]
	                    [--readout READOUT] [--slice_drop_correction]
	                    [--out_directory OUT_DIRECTORY] [--out_prefix OUT_PREFIX]
	                    [--out_script] [--fix_seed] [-f] [-v]
	                    in_dwi in_bvals in_bvecs in_mask
	
	Prepare a typical command for eddy and create the necessary files.
	
	positional arguments:
	  in_dwi                input DWI Nifti image
	  in_bvals              b-values file in FSL format
	  in_bvecs              b-vectors file in FSL format
	  in_mask               binary brain mask.
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --topup TOPUP         topup output name. If given, apply topup during eddy.
	                        Should be the same as --out_prefix from scil_prepare_topup_command.py
	  --eddy_cmd {eddy_openmp,eddy_cuda}
	                        eddy command [eddy_openmp].
	  --b0_thr B0_THR       All b-values with values less than or equal to b0_thr are considered
	                        as b0s i.e. without diffusion weighting
	  --encoding_direction {x,y,z}
	                        acquisition direction, default is AP-PA [y].
	  --readout READOUT     total readout time from the DICOM metadata [0.062].
	  --slice_drop_correction
	                        if set, will activate eddy's outlier correction,
	                        which includes slice drop correction.
	  --out_directory OUT_DIRECTORY
	                        output directory for eddy files [.].
	  --out_prefix OUT_PREFIX
	                        prefix of the eddy-corrected DWI [dwi_eddy_corrected].
	  --out_script          if set, will output a .sh script (eddy.sh).
	                        else, will output the lines to the terminal [False].
	  --fix_seed            if set, will use the fixed seed strategy for eddy.
	                        Enhances reproducibility.
	  -f                    Force overwriting of the output files.
	  -v                    If set, produces verbose output.
