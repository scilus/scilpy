scil_merge_sh.py
==============

::

	usage: scil_merge_sh.py [-h] [-f] sh_files [sh_files ...] out_sh
	
	Merge a list of Spherical Harmonics files.
	
	This merges the coefficients of multiple Spherical Harmonics files
	by taking, for each coefficient, the one with the largest magnitude.
	
	Can be used to merge fODFs computed from different shells into 1, while
	conserving the most relevant information.
	
	Based on [1].
	
	positional arguments:
	  sh_files    List of SH files.
	  out_sh      output SH file.
	
	optional arguments:
	  -h, --help  show this help message and exit
	  -f          Force overwriting of the output files.
	
	Reference:
	    [1] Garyfallidis, E., Zucchelli, M., Houde, J-C., Descoteaux, M.
	        How to perform best ODF reconstruction from the Human Connectome
	        Project sampling scheme?
	        ISMRM 2014.
