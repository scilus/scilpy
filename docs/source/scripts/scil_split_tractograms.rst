scil_split_tractograms.py
==============

::

	usage: scil_split_tractograms.py [-h] (--chunk_size CHUNK_SIZE | --nb_chunk NB_CHUNK)
	                    [--reference REFERENCE] [-f]
	                    in_tractogram out_tractogram
	
	Split a tractogram into multiple files, 2 options available :
	Split into X files, or split into files of Y streamlines
	
	positional arguments:
	  in_tractogram         Tractogram input file name.
	  out_tractogram        Output filename, with extension needed,index will be appended automatically.
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --chunk_size CHUNK_SIZE
	                        The maximum number of streamlines per file.
	  --nb_chunk NB_CHUNK   Divide the file in equal parts.
	  --reference REFERENCE
	                        Reference anatomy for tck/vtk/fib/dpy file
	                        support (.nii or .nii.gz).
	  -f                    Force overwriting of the output files.
