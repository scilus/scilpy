scil_reorder_dwi_philips.py
==============

::

	usage: scil_reorder_dwi_philips.py [-h] [-f] [-v] dwi bvec bval table baseName
	
	Re-order gradient according to original table
	
	positional arguments:
	  dwi         input dwi file
	  bvec        input bvec FSL format
	  bval        input bval FSL format
	  table       original table - first line is skipped
	  baseName    basename output file
	
	optional arguments:
	  -h, --help  show this help message and exit
	  -f          Force overwriting of the output files.
	  -v          If set, produces verbose output.
