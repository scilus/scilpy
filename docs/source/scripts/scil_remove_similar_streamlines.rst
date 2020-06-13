scil_remove_similar_streamlines.py
==============

::

	usage: scil_remove_similar_streamlines.py [-h] [--clustering_thr CLUSTERING_THR]
	                    [--min_cluster_size MIN_CLUSTER_SIZE]
	                    [--convergence CONVERGENCE] [--avg_similar]
	                    [--processes PROCESSES] [--reference REFERENCE] [-f] [-v]
	                    in_bundle min_distance out_bundle
	
	Remove very similar streamlines from a bundle.
	Uses clustering to speed up the process. Streamlines are considered as similar
	based on a MDF threshold within each cluster. Can be used with large bundles,
	but the clustering parameters will need to be adjusted.
	
	The algorithm still uses a system of chunks to ensure the amount of comparison
	(n**2) does not grow out of control. To overcome limitations related to this
	use of chunks, multiple iterations must be done until a convergence threshold
	is achieved.
	
	The subsampling threshold should be between 2mm and 5mm, 5mm being quite
	aggressive. A CST where all fanning streamlines are important should be around
	2mm, while an AF can go around 4mm.
	
	The --processes parameters should only be use on massive bundle. For example,
	100 000 streamlines can be split among 8 processes.
	
	positional arguments:
	  in_bundle             Path of the input bundle.
	  min_distance          Distance threshold for 2 streamlines to be considered similar (mm).
	  out_bundle            Path of the output tractography file
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --clustering_thr CLUSTERING_THR
	                        Clustering threshold for QB/QBx (mm), during the first approximation [6].
	  --min_cluster_size MIN_CLUSTER_SIZE
	                        Minimum cluster size for the first iteration [5].
	  --convergence CONVERGENCE
	                        Streamlines count difference threshold to stop re-running the algorithm [100].
	  --avg_similar         Average similar streamlines rather than removing them [False]. Requires a small min_distance. Allows for some smoothing.
	  --processes PROCESSES
	                        Number of desired processes [1].
	  --reference REFERENCE
	                        Reference anatomy for tck/vtk/fib/dpy file
	                        support (.nii or .nii.gz).
	  -f                    Force overwriting of the output files.
	  -v                    If set, produces verbose output.
