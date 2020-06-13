scil_outlier_rejection.py
==============

::

	usage: scil_outlier_rejection.py [-h] [--remaining_bundle REMAINING_BUNDLE] [--alpha ALPHA]
	                    [-f]
	                    in_bundle out_bundle
	
	Clean a bundle (inliers/outliers) using hiearchical clustering.
	http://archive.ismrm.org/2015/2844.html
	
	If spurious streamlines are dense, it is possible they will not be recognized
	as outliers. Manual cleaning may be required to overcome this limitation.
	
	positional arguments:
	  in_bundle             Fiber bundle file to remove outliers from.
	  out_bundle            Fiber bundle without outliers.
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --remaining_bundle REMAINING_BUNDLE
	                        Removed outliers.
	  --alpha ALPHA         Percent of the length of the tree that clusters of individual streamlines will be pruned.
	  -f                    Force overwriting of the output files.
