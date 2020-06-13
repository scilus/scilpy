scil_extract_dwi_shell.py
==============

::

	usage: scil_extract_dwi_shell.py [-h] [--block-size INT] [--tolerance INT] [-v] [-f]
	                    dwi bvals bvecs bvals-to-extract [bvals-to-extract ...]
	                    output_dwi output_bvals output_bvecs
	
	Extracts the DWI volumes that are on specific b-value shells. Many shells
	can be extracted at once by specifying multiple b-values. The extracted
	volumes are in the same order as in the original file.
	
	If the b-values of a shell are not all identical, use the --tolerance
	argument to adjust the accepted interval. For example, a b-value of 2000
	and a tolerance of 20 will extract all volumes with a b-values from 1980 to
	2020.
	
	Files that are too large to be loaded in memory can still be processed by
	setting the --block-size argument. A block size of X means that X DWI volumes
	are loaded at a time for processing.
	
	positional arguments:
	  dwi                   The DW image file to split.
	  bvals                 The b-values in FSL format.
	  bvecs                 The b-vectors in FSL format.
	  bvals-to-extract      The list of b-values to extract. For example 0 2000.
	  output_dwi            The name of the output DWI file.
	  output_bvals          The name of the output b-values.
	  output_bvecs          The name of the output b-vectors
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --block-size INT, -s INT
	                        Loads the data using this block size. Useful
	                        when the data is too large to be loaded in memory.
	  --tolerance INT, -t INT
	                        The tolerated gap between the b-values to extract
	                        and the actual b-values.
	  -v                    If set, produces verbose output.
	  -f                    Force overwriting of the output files.
