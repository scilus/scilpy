scil_remove_labels.py
==============

::

	usage: scil_remove_labels.py [-h] -i INDICES [INDICES ...] [--background BACKGROUND]
	                    [-f]
	                    in_labels out_labels
	
	    Script to remove specific labels from a atlas volumes.
	
	    >>> scil_remove_labels.py DKT_labels.nii out_labels.nii.gz -i 5001 5002
	
	positional arguments:
	  in_labels             Input labels volume.
	  out_labels            Output labels volume.
	
	optional arguments:
	  -h, --help            show this help message and exit
	  -i INDICES [INDICES ...], --indices INDICES [INDICES ...]
	                        List of labels indices to remove.
	  --background BACKGROUND
	                        Integer used for removed labels [0].
	  -f                    Force overwriting of the output files.
	
	    References:
	        [1] Al-Sharif N.B., St-Onge E., Vogel J.W., Theaud G.,
	            Evans A.C. and Descoteaux M. OHBM 2019.
	            Surface integration for connectome analysis in age prediction.
	    
