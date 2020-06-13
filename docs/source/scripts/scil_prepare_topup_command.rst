scil_prepare_topup_command.py
==============

::

	usage: scil_prepare_topup_command.py [-h] [--config CONFIG] [--b0_thr B0_THR]
	                    [--encoding_direction {x,y,z}] [--readout READOUT]
	                    [--out_b0s OUT_B0S] [--out_directory OUT_DIRECTORY]
	                    [--out_prefix OUT_PREFIX] [--out_script] [-f] [-v]
	                    in_dwi in_bvals in_bvecs in_reverse_b0
	
	Prepare a typical command for topup and create the necessary files.
	The reversed b0 must be in a different file.
	
	positional arguments:
	  in_dwi                input DWI Nifti image
	  in_bvals              b-values file in FSL format
	  in_bvecs              b-vectors file in FSL format
	  in_reverse_b0         b0 image with reversed phase encoding.
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --config CONFIG       topup config file [b02b0.cnf].
	  --b0_thr B0_THR       All b-values with values less than or equal to b0_thr are considered as b0s i.e. without diffusion weighting
	  --encoding_direction {x,y,z}
	                        acquisition direction, default is AP-PA [y].
	  --readout READOUT     total readout time from the DICOM metadata [0.062].
	  --out_b0s OUT_B0S     output fused b0 file [fused_b0s.nii.gz].
	  --out_directory OUT_DIRECTORY
	                        output directory for topup files [.].
	  --out_prefix OUT_PREFIX
	                        prefix of the topup results [topup_results].
	  --out_script          if set, will output a .sh script (topup.sh).
	                        else, will output the lines to the terminal [False].
	  -f                    Force overwriting of the output files.
	  -v                    If set, produces verbose output.
