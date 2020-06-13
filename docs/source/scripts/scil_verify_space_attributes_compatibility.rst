scil_verify_space_attributes_compatibility.py
==============

::

	usage: scil_verify_space_attributes_compatibility.py [-h] in_files [in_files ...]
	
	Will compare all input files against the first one for the compatibility
	of their spatial attributes.
	
	Spatial attributes are: affine, dimensions, voxel sizes and voxel order.
	
	positional arguments:
	  in_files    List of file to compare (trk and nii).
	
	optional arguments:
	  -h, --help  show this help message and exit
