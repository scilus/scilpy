scil_clean_qbx_clusters.py
==============

::

	usage: scil_clean_qbx_clusters.py [-h] [--out_accepted_dir OUT_ACCEPTED_DIR]
	                    [--out_rejected_dir OUT_REJECTED_DIR]
	                    [--min_cluster_size MIN_CLUSTER_SIZE]
	                    [--background_opacity BACKGROUND_OPACITY]
	                    [--background_linewidth BACKGROUND_LINEWIDTH]
	                    [--clusters_linewidth CLUSTERS_LINEWIDTH]
	                    [--reference REFERENCE] [-f] [-v]
	                    in_bundles [in_bundles ...] out_accepted out_rejected
	
	    Render clusters sequentially to either accept or reject them based on
	    visual inspection. Useful for cleaning bundles for RBx, BST or for figures.
	    The VTK window does not handle well opacity of streamlines, this is a normal
	    rendering behavior.
	    Often use in pair with scil_compute_qbx.py.
	
	    Key mapping:
	    - a/A: accept displayed clusters
	    - r/R: reject displayed clusters
	    - z/Z: Rewing one element
	    - c/C: Stop rendering of the background concatenation of streamlines
	    - q/Q: Early window exist, everything remaining will be rejected
	
	positional arguments:
	  in_bundles            List of the clusters filename.
	  out_accepted          Filename of the concatenated accepted clusters.
	  out_rejected          Filename of the concatenated rejected clusters.
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --out_accepted_dir OUT_ACCEPTED_DIR
	                        Directory to save all accepted clusters separately.
	  --out_rejected_dir OUT_REJECTED_DIR
	                        Directory to save all rejected clusters separately.
	  --min_cluster_size MIN_CLUSTER_SIZE
	                        Minimum cluster size for consideration [1].Must be at least 1.
	  --background_opacity BACKGROUND_OPACITY
	                        Opacity of the background streamlines.Keep low between 0 and 0.5 [0.1].
	  --background_linewidth BACKGROUND_LINEWIDTH
	                        Linewidth of the background streamlines [1].
	  --clusters_linewidth CLUSTERS_LINEWIDTH
	                        Linewidth of the current cluster [1].
	  --reference REFERENCE
	                        Reference anatomy for tck/vtk/fib/dpy file
	                        support (.nii or .nii.gz).
	  -f                    Force overwriting of the output files.
	  -v                    If set, produces verbose output.
