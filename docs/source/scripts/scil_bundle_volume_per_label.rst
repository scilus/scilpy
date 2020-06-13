scil_bundle_volume_per_label.py
==============

::

	usage: scil_bundle_volume_per_label.py [-h] [--indent INDENT] [--sort_keys] [-f]
	                    voxel_label_map bundle_name
	
	Compute bundle volume per label in mm³. This script supports anisotropic voxels
	resolution. Volume is estimated by counting the number of voxel occupied by
	each label and multiplying it by the volume of a single voxel.
	
	This estimation is typically performed at resolution around 1mm³.
	
	positional arguments:
	  voxel_label_map  Fiber bundle file.
	  bundle_name      Bundle name.
	
	optional arguments:
	  -h, --help       show this help message and exit
	  -f               Force overwriting of the output files.
	
	Json options:
	  --indent INDENT  Indent for json pretty print.
	  --sort_keys      Sort keys in output json.
