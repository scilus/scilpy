scil_compute_seed_density_map.py
==============

::

	usage: scil_compute_seed_density_map.py [-h] [--binary [FIXED_VALUE]] [-f]
	                    tractogram_filename seed_density_filename
	
	Compute a density map of seeds saved in .trk file.
	
	positional arguments:
	  tractogram_filename   Tracts filename. Format must be .trk.
	  seed_density_filename
	                        Output seed density filename. Format must be Nifti.
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --binary [FIXED_VALUE]
	                        If set, will store the same value for all intersected voxels, creating a binary map.
	                        When set without a value, 1 is used.
	                         If a value is given, will be used as the stored value.
	  -f                    Force overwriting of the output files.
