scil_count_non_zero_voxels.py
==============

::

	usage: scil_count_non_zero_voxels.py [-h] [--out OUT_FILE] [--stats] [--id VALUE_ID] IN_FILE
	
	Count the number of non-zero voxels in an image file.
	
	If you give it an image with more than 3 dimensions, it will summarize the 4th
	(or more) dimension to one voxel, and then find non-zero voxels over this.
	This means that if there is at least one non-zero voxel in the 4th dimension,
	this voxel of the 3D volume will be considered as non-zero.
	
	positional arguments:
	  IN_FILE         input file name, in nifti format.
	
	optional arguments:
	  -h, --help      show this help message and exit
	  --out OUT_FILE  name of the output file, which will be saved as a text file.
	  --stats         output the value using a stats format. Using this syntax will append
	                  a line to the output file, instead of creating a file with only one line.
	                  This is useful to create a file to be used as the source of data for a graph.
	                  Can be combined with --id
	  --id VALUE_ID   Id of the current count. If used, the value of this argument will be
	                  output (followed by a ":") before the count value.
	                  Mostly useful with --stats.
