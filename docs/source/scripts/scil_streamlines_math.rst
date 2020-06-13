scil_streamlines_math.py
==============

::

	usage: scil_streamlines_math.py [-h] [--precision NUMBER_OF_DECIMALS] [--no_metadata]
	                    [--save_metadata_indices]
	                    [--save_indices OUTPUT_INDEX_FILE] [--reference REFERENCE]
	                    [-v] [-f]
	                    OPERATION INPUT_FILES [INPUT_FILES ...] OUTPUT_FILE
	
	Performs an operation on a list of streamline files. The supported
	operations are:
	
	    difference:  Keep the streamlines from the first file that are not in
	                 any of the following files.
	
	    intersection: Keep the streamlines that are present in all files.
	
	    union:        Keep all streamlines while removing duplicates.
	
	    concatenate:  Keep all streamlines with duplicates.
	
	For efficiency, the comparisons are performed using a hash table. This means
	that streamlines must be identical for a match to be found. To allow a soft
	match, use the --precision option to round streamlines before processing.
	Note that the streamlines that are saved in the output are the original
	streamlines, not the rounded ones.
	
	The metadata (data per point, data per streamline) of the streamlines that
	are kept in the output will preserved. This requires that all input files
	share the same type of metadata. If this is not the case, use the option
	--no-data to strip the metadata from the output.
	
	Repeated uses with .trk files will slighly affect coordinate values
	due to precision error.
	
	positional arguments:
	  OPERATION             The type of operation to be performed on the streamlines. Must
	                        be one of the following: difference, intersection, union, concatenate.
	  INPUT_FILES           The list of files that contain the streamlines to operate on.
	  OUTPUT_FILE           The file where the remaining streamlines are saved.
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --precision NUMBER_OF_DECIMALS, -p NUMBER_OF_DECIMALS
	                        The precision used when comparing streamlines.
	  --no_metadata, -n     Strip the streamline metadata from the output.
	  --save_metadata_indices, -m
	                        Save streamline indices to metadata. Has no effect if --no-data
	                        is present. Will overwrite 'ids' metadata if already present.
	  --save_indices OUTPUT_INDEX_FILE, -s OUTPUT_INDEX_FILE
	                        Save the streamline indices to the supplied json file.
	  --reference REFERENCE
	                        Reference anatomy for tck/vtk/fib/dpy file
	                        support (.nii or .nii.gz).
	  -v                    If set, produces verbose output.
	  -f                    Force overwriting of the output files.
