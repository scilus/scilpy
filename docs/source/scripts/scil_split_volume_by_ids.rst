scil_split_volume_by_ids.py
==============

::

	usage: scil_split_volume_by_ids.py [-h] [--out_dir OUT_DIR] [--out_prefix OUT_PREFIX]
	                    [-r [RANGE [RANGE ...]]] [-f]
	                    in_labels
	
	Split a label image into multiple images where the name of the output images
	is the id of the label (ex. 35.nii.gz, 36.nii.gz, ...). If the --range option
	is not provided, all labels of the image are extracted.
	
	IMPORTANT: your label image must be of an integer type.
	
	positional arguments:
	  in_labels             Path of the input label file, in a format supported by Nibabel.
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --out_dir OUT_DIR     Put all ouptput images in a specific directory.
	  --out_prefix OUT_PREFIX
	                        Prefix to be used for each output image.
	  -r [RANGE [RANGE ...]], --range [RANGE [RANGE ...]]
	                        Specifies a subset of labels to split, formatted as 1-3 or 3 4.
	  -f                    Force overwriting of the output files.
