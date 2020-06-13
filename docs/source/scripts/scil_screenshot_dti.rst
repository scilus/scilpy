scil_screenshot_dti.py
==============

::

	usage: scil_screenshot_dti.py [-h] [--shells SHELLS [SHELLS ...]]
	                    [--output_suffix OUTPUT_SUFFIX] [--output_dir OUTPUT_DIR]
	                    [-f]
	                    dwi bval bvec target_template
	
	Register DWI to a template for screenshots.
	The templates are on http://www.bic.mni.mcgill.ca/ServicesAtlases/ICBM152NLin2009
	
	For quick quality control, the MNI template can be downsampled to 2mm iso.
	Axial, coronal and sagittal slices are captured.
	
	positional arguments:
	  dwi                   Path of the input diffusion volume.
	  bval                  Path of the bval file, in FSL format.
	  bvec                  Path of the bvec file, in FSL format.
	  target_template       Path to the target MNI152 template for registration,
	                        use the one provided online.
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --shells SHELLS [SHELLS ...]
	                        Shells to use for DTI fit (usually below 1200), b0 must be listed.
	  --output_suffix OUTPUT_SUFFIX
	                        Add a suffix to the output, else the axis name is used.
	  --output_dir OUTPUT_DIR
	                        Put all images in a specific directory.
	  -f                    Force overwriting of the output files.
