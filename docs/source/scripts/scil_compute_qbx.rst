scil_compute_qbx.py
==============

::

	usage: scil_compute_qbx.py [-h] [--nb_points NB_POINTS]
	                    [--output_centroids OUTPUT_CENTROIDS]
	                    [--reference REFERENCE] [-f]
	                    in_tractogram dist_thresh output_clusters_dir
	
	    Compute clusters using QuickBundlesX and save them separately.
	    We cannot know the number of clusters in advance.
	
	positional arguments:
	  in_tractogram         Tractogram filename.
	                        Path of the input tractogram or bundle.
	  dist_thresh           Last QuickBundlesX threshold in mm. Typically 
	                        the value are between 10-20mm.
	  output_clusters_dir   Path to the clusters directory.
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --nb_points NB_POINTS
	                        Streamlines will be resampled to have this number of points [20].
	  --output_centroids OUTPUT_CENTROIDS
	                        Output tractogram filename.
	                        Format must be readable by the Nibabel API.
	  --reference REFERENCE
	                        Reference anatomy for tck/vtk/fib/dpy file
	                        support (.nii or .nii.gz).
	  -f                    Force overwriting of the output files.
