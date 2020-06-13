scil_dilate_labels.py
==============

::

	usage: scil_dilate_labels.py [-h] [--distance DISTANCE]
	                    [--label_to_dilate LABEL_TO_DILATE [LABEL_TO_DILATE ...]]
	                    [--label_to_fill LABEL_TO_FILL [LABEL_TO_FILL ...]]
	                    [--label_not_to_dilate LABEL_NOT_TO_DILATE [LABEL_NOT_TO_DILATE ...]]
	                    [--mask MASK] [--processes NBR] [-f]
	                    in_file out_file
	
	Dilate regions (with or without masking) from a labeled volume:
	- "label_to_dilate" are regions that will dilate over
	    "label_to_fill" if close enough to it ("distance").
	- "label_to_dilate", by default (None) will be all
	        non-"label_to_fill" and non-"label_not_to_dilate".
	- "label_not_to_dilate" will not be changed, but will not dilate.
	- "mask" is where the dilation is allowed (constrained)
	    in addition to "background_label" (logical AND)
	
	>>> scil_dilate_labels.py wmparc_t1.nii.gz wmparc_dil.nii.gz \
	    --label_to_fill 0 5001 5002 \
	    --label_not_to_dilate 4 43 10 11 12 49 50 51
	
	positional arguments:
	  in_file               Path of the volume (nii or nii.gz).
	  out_file              Output filename of the dilated labels.
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --distance DISTANCE   Maximal distance to dilated (in mm) [2.0].
	  --label_to_dilate LABEL_TO_DILATE [LABEL_TO_DILATE ...]
	                        Label list to dilate, by default it dilate all that
	                         are not in label_to_fill nor label_not_to_dilate.
	  --label_to_fill LABEL_TO_FILL [LABEL_TO_FILL ...]
	                        Background id / labels to be filled [[0]],
	                         the first one is given as output background value.
	  --label_not_to_dilate LABEL_NOT_TO_DILATE [LABEL_NOT_TO_DILATE ...]
	                        Label list not to dilate.
	  --mask MASK           Only dilate values inside the mask.
	  --processes NBR       Number of sub-processes to start. 
	                        Default: [1]
	  -f                    Force overwriting of the output files.
	
	    References:
	        [1] Al-Sharif N.B., St-Onge E., Vogel J.W., Theaud G.,
	            Evans A.C. and Descoteaux M. OHBM 2019.
	            Surface integration for connectome analysis in age prediction.
	    
