scil_screenshot_bundle.py
==============

::

	usage: scil_screenshot_bundle.py [-h] [--target_template TARGET_TEMPLATE]
	                    [--local_coloring | --uniform_coloring R G B | --reference_coloring COLORBAR]
	                    [--right] [--anat_opacity ANAT_OPACITY]
	                    [--output_suffix OUTPUT_SUFFIX] [--output_dir OUTPUT_DIR]
	                    [-v] [-f]
	                    in_bundle in_anat
	
	Register bundle to a template for screenshots using a reference.
	The template can be any MNI152 (any resolution, cropped or not)
	If your in_anat has a skull, select a MNI152 template with a skull and
	vice-versa.
	
	If the bundle is already in MNI152 space, do not use --target_template.
	
	Axial, coronal and sagittal slices are captured.
	Sagittal can be capture from the left (default) or the right.
	
	positional arguments:
	  in_bundle             Path of the input bundle.
	  in_anat               Path of the reference file (.nii or nii.gz).
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --target_template TARGET_TEMPLATE
	                        Path to the target MNI152 template for registration. 
	                        If in_anat has a skull, select a MNI152 template 
	                        with a skull and vice-versa.
	  --local_coloring      Color streamlines local segments orientation.
	  --uniform_coloring R G B
	                        Color streamlines with uniform coloring.
	  --reference_coloring COLORBAR
	                        Color streamlines with reference coloring (0-255).
	  --right               Take screenshot from the right instead of the left 
	                        for the sagittal plane.
	  --anat_opacity ANAT_OPACITY
	                        Set the opacity for the anatomy, use 0 for complete 
	                        transparency, 1 for opaque.
	  --output_suffix OUTPUT_SUFFIX
	                        Add a suffix to the output, else the axis name is used.
	  --output_dir OUTPUT_DIR
	                        Put all images in a specific directory.
	  -v                    If set, produces verbose output.
	  -f                    Force overwriting of the output files.
