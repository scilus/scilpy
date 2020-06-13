scil_add_tracking_mask_to_pft_maps.py
==============

::

	usage: scil_add_tracking_mask_to_pft_maps.py [-h] [-f]
	                    map_include map_exclude additional_mask map_include_corr
	                    map_exclude_corr
	
	Modify PFT maps to allow PFT tracking in given mask (e.g edema).
	
	positional arguments:
	  map_include       PFT map include.
	  map_exclude       PFT map exclude.
	  additional_mask   Allow PFT tracking in this mask.
	  map_include_corr  Corrected PFT map include output file name.
	  map_exclude_corr  Corrected PFT map exclude output file name.
	
	optional arguments:
	  -h, --help        show this help message and exit
	  -f                Force overwriting of the output files.
