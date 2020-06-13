scil_image_math.py
==============

::

	usage: scil_image_math.py [-h] [--data_type DATA_TYPE] [--exclude_background] [-f]
	                    [-v]
	                    {lower_threshold,upper_threshold,lower_clip,upper_clip,absolute_value,round,ceil,floor,normalize_sum,normalize_max,log_10,log_e,convert,invert,addition,subtraction,multiplication,division,mean,std,union,intersection,difference,dilation,erosion,closing,opening,blur}
	                    in_images [in_images ...] out_image
	
	Performs an operation on a list of images. The supported operations are
	listed below.
	
	This script is loading all images in memory, will often crash after a few
	hundred images.
	
	Some operations such as multiplication or addition accept float value as
	parameters instead of images.
	> scil_image_math.py multiplication img.nii.gz 10 mult_10.nii.gz
	
	    lower_threshold: IMG THRESHOLD
	        All values below the threshold will be set to zero.
	        All values above the threshold will be set to one.
	    
	    upper_threshold: IMG THRESHOLD
	        All values below the threshold will be set to one.
	        All values above the threshold will be set to zero.
	        Equivalent to lower_threshold followed by an inversion.
	    
	    lower_clip: IMG THRESHOLD
	        All values below the threshold will be set to threshold.
	    
	    upper_clip: IMG THRESHOLD
	        All values above the threshold will be set to threshold.
	    
	    absolute_value: IMG
	        All negative values will become positive.
	    
	    round: IMG
	        Round all decimal values to the closest integer.
	    
	    ceil: IMG
	        Ceil all decimal values to the next integer.
	    
	    floor: IMG
	        Floor all decimal values to the previous integer.
	    
	    normalize_sum: IMG
	        Normalize the image so the sum of all values is one.
	    
	    normalize_max: IMG
	        Normalize the image so the maximum value is one.
	    
	    log_10: IMG
	        Apply a log (base 10) to all non zeros values of an image.
	    
	    log_e: IMG
	        Apply a natural log to all non zeros values of an image.
	    
	    convert: IMG
	        Perform no operation, but simply change the data type.
	    
	    invert: IMG
	        Operation on binary image to interchange 0s and 1s in a binary mask.
	    
	    addition: IMGs
	        Add multiple images together.
	    
	    subtraction: IMG_1 IMG_2
	        Subtract first image by the second (IMG_1 - IMG_2).
	    
	    multiplication: IMGs
	        Multiply multiple images together (danger of underflow and overflow)
	    
	    division: IMG_1 IMG_2
	        Divide first image by the second (danger of underflow and overflow)
	        Ignore zeros values, excluded from the operation.
	    
	    mean: IMGs
	        Compute the mean of images.
	        If a single 4D image is provided, average along the last dimension.
	    
	    std: IMGs
	        Compute the standard deviation average of multiple images.
	        If a single 4D image is provided, compute the STD along the last
	        dimension.
	    
	    union: IMGs
	        Operation on binary image to keep voxels, that are non-zero, in at
	        least one file.
	    
	    intersection: IMGs
	        Operation on binary image to keep the voxels, that are non-zero,
	        are present in all files.
	    
	    difference: IMG_1 IMG_2
	        Operation on binary image to keep voxels from the first file that are
	        not in the second file (non-zeros).
	    
	    dilation: IMG, VALUE
	        Binary morphological operation to spatially extend the values of an
	        image to their neighbors.
	    
	    erosion: IMG, VALUE
	        Binary morphological operation to spatially shrink the volume contained
	        in a binary image.
	    
	    closing: IMG, VALUE
	        Binary morphological operation, dilation followed by an erosion.
	    
	    opening: IMG, VALUE
	        Binary morphological operation, erosion followed by a dilation.
	    
	    blur: IMG, VALUE
	        Apply a gaussian blur to a single image.
	    
	
	positional arguments:
	  {lower_threshold,upper_threshold,lower_clip,upper_clip,absolute_value,round,ceil,floor,normalize_sum,normalize_max,log_10,log_e,convert,invert,addition,subtraction,multiplication,division,mean,std,union,intersection,difference,dilation,erosion,closing,opening,blur}
	                        The type of operation to be performed on the images.
	  in_images             The list of image files or parameters.
	  out_image             Output image path.
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --data_type DATA_TYPE
	                        Data type of the output image. Use the format: uint8, int16, int/float32, int/float64.
	  --exclude_background  Does not affect the background of the original image.
	  -f                    Force overwriting of the output files.
	  -v                    If set, produces verbose output.
