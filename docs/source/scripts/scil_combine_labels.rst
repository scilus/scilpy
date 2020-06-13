scil_combine_labels.py
==============

::

	usage: scil_combine_labels.py [-h] -v VOLUME_IDS [VOLUME_IDS ...]
	                    [--out_labels_ids OUT_LABELS_IDS [OUT_LABELS_IDS ...] |
	                    --unique | --group_in_m] [--background BACKGROUND] [-f]
	                    output
	
	    Script to combine labels from multiple volumes,
	        if there is overlap, it will overwrite them based on the input order.
	
	    >>> scil_combine_labels.py out_labels.nii.gz  -v animal_labels.nii 20\
	            DKT_labels.nii.gz 44 53  --out_labels_indices 20 44 53
	    >>> scil_combine_labels.py slf_labels.nii.gz  -v a2009s_aseg.nii.gz all\
	            -v clean/s1__DKT.nii.gz 1028 2028
	
	positional arguments:
	  output                Combined labels volume output.
	
	optional arguments:
	  -h, --help            show this help message and exit
	  -v VOLUME_IDS [VOLUME_IDS ...], --volume_ids VOLUME_IDS [VOLUME_IDS ...]
	                        List of volumes directly followed by their labels:
	                          -v atlasA  id1a id2a   -v  atlasB  id1b id2b ... 
	                          "all" can be used instead of id numbers.
	  --out_labels_ids OUT_LABELS_IDS [OUT_LABELS_IDS ...]
	                        Give a list of labels indices for output images.
	  --unique              Output id with unique labels, excluding first background value.
	  --group_in_m          Add (x*1000000) to each volume labels, where x is the input volume order number.
	  --background BACKGROUND
	                        Background id, excluded from output [0],
	                         the value is used as output background value.
	  -f                    Force overwriting of the output files.
	
	    References:
	        [1] Al-Sharif N.B., St-Onge E., Vogel J.W., Theaud G.,
	            Evans A.C. and Descoteaux M. OHBM 2019.
	            Surface integration for connectome analysis in age prediction.
	    
