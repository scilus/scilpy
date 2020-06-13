scil_connectivity_math.py
==============

::

	usage: scil_connectivity_math.py [-h] [--data_type DATA_TYPE] [-f] [-v]
	                    {lower_threshold,upper_threshold,lower_clip,upper_clip,absolute_value,round,ceil,floor,normalize_sum,normalize_max,log_10,log_e,convert,invert,addition,subtraction,multiplication,division,mean,std,union,intersection,difference}
	                    inputs [inputs ...] out_matrix
	
	Performs an operation on a list of matrices. The supported operations are
	listed below.
	
	Some operations such as multiplication or addition accept float value as
	parameters instead of matrices.
	> scil_connectivity_math.py multiplication mat.npy 10 mult_10.npy
	
	    lower_threshold: MAT THRESHOLD
	        All values below the threshold will be set to zero.
	        All values above the threshold will be set to one.
	    
	    upper_threshold: MAT THRESHOLD
	        All values below the threshold will be set to one.
	        All values above the threshold will be set to zero.
	        Equivalent to lower_threshold followed by an inversion.
	    
	    lower_clip: MAT THRESHOLD
	        All values below the threshold will be set to threshold.
	    
	    upper_clip: MAT THRESHOLD
	        All values above the threshold will be set to threshold.
	    
	    absolute_value: MAT
	        All negative values will become positive.
	    
	    round: MAT
	        Round all decimal values to the closest integer.
	    
	    ceil: MAT
	        Ceil all decimal values to the next integer.
	    
	    floor: MAT
	        Floor all decimal values to the previous integer.
	    
	    normalize_sum: MAT
	        Normalize the matrix so the sum of all values is one.
	    
	    normalize_max: MAT
	        Normalize the matrix so the maximum value is one.
	    
	    log_10: MAT
	        Apply a log (base 10) to all non zeros values of an matrix.
	    
	    log_e: MAT
	        Apply a natural log to all non zeros values of an matrix.
	    
	    convert: MAT
	        Perform no operation, but simply change the data type.
	    
	    invert: MAT
	        Operation on binary matrix to interchange 0s and 1s in a binary mask.
	    
	    addition: MATs
	        Add multiple matrices together.
	    
	    subtraction: MAT_1 MAT_2
	        Subtract first matrix by the second (MAT_1 - MAT_2).
	    
	    multiplication: MATs
	        Multiply multiple matrices together (danger of underflow and overflow)
	    
	    division: MAT_1 MAT_2
	        Divide first matrix by the second (danger of underflow and overflow)
	        Ignore zeros values, excluded from the operation.
	    
	    mean: MATs
	        Compute the mean of matrices.
	        If a single 4D matrix is provided, average along the last dimension.
	    
	    std: MATs
	        Compute the standard deviation average of multiple matrices.
	        If a single 4D matrix is provided, compute the STD along the last
	        dimension.
	    
	    union: MATs
	        Operation on binary matrix to keep voxels, that are non-zero, in at
	        least one file.
	    
	    intersection: MATs
	        Operation on binary matrix to keep the voxels, that are non-zero,
	        are present in all files.
	    
	    difference: MAT_1 MAT_2
	        Operation on binary matrix to keep voxels from the first file that are
	        not in the second file (non-zeros).
	    
	
	positional arguments:
	  {lower_threshold,upper_threshold,lower_clip,upper_clip,absolute_value,round,ceil,floor,normalize_sum,normalize_max,log_10,log_e,convert,invert,addition,subtraction,multiplication,division,mean,std,union,intersection,difference}
	                        The type of operation to be performed on the matrices.
	  inputs                The list of matrices files or parameters.
	  out_matrix            Output matrix path.
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --data_type DATA_TYPE
	                        Data type of the output image. Use the format: uint8, float16, int32.
	  -f                    Force overwriting of the output files.
	  -v                    If set, produces verbose output.
