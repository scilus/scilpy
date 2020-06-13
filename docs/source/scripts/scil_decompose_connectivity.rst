scil_decompose_connectivity.py
==============

::

	usage: scil_decompose_connectivity.py [-h] [--no_pruning] [--no_remove_loops]
	                    [--no_remove_outliers] [--no_remove_curv_dev]
	                    [--min_length MIN_LENGTH] [--max_length MAX_LENGTH]
	                    [--outlier_threshold OUTLIER_THRESHOLD]
	                    [--loop_max_angle LOOP_MAX_ANGLE]
	                    [--curv_qb_distance CURV_QB_DISTANCE] [--out_dir OUT_DIR]
	                    [--save_raw_connections] [--save_intermediate]
	                    [--save_discarded] [--out_labels_list OUT_FILE]
	                    [--reference REFERENCE] [-v] [-f]
	                    in_tractogram in_labels out_hdf5
	
	Compute a connectivity matrix from a tractogram and a parcellation.
	
	Current strategy is to keep the longest streamline segment connecting
	2 regions. If the streamline crosses other gray matter regions before
	reaching its final connected region, the kept connection is still the
	longest. This is robust to compressed streamlines.
	
	The output file is a hdf5 (.h5) where the keys are 'LABEL1_LABEL2' and each
	group is composed of 'data', 'offsets' and 'lengths' from the array_sequence.
	The 'data' is stored in VOX/CORNER for simplicity and efficiency.
	
	NOTE: this script can take a while to run. Please be patient.
	Example: on a tractogram with 1.8M streamlines, running on a SSD:
	- 15 minutes without post-processing, only saving final bundles.
	- 30 minutes with full post-processing, only saving final bundles.
	- 60 minutes with full post-processing, saving all possible files.
	
	positional arguments:
	  in_tractogram         Tractogram filename. Format must be one of 
	                        trk, tck, vtk, fib, dpy.
	  in_labels             Labels file name (nifti). Labels must have 0 as background.
	  out_hdf5              Output hdf5 file (.h5).
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --out_labels_list OUT_FILE
	                        Save the labels list as text file.
	                        Needed for scil_compute_connectivity.py and others.
	  --reference REFERENCE
	                        Reference anatomy for tck/vtk/fib/dpy file
	                        support (.nii or .nii.gz).
	  -v                    If set, produces verbose output.
	  -f                    Force overwriting of the output files.
	
	Post-processing options:
	  --no_pruning          If set, will NOT prune on length.
	                        Length criteria in --min_length, --max_length.
	  --no_remove_loops     If set, will NOT remove streamlines making loops.
	                        Angle criteria based on --loop_max_angle.
	  --no_remove_outliers  If set, will NOT remove outliers using QB.
	                        Criteria based on --outlier_threshold.
	  --no_remove_curv_dev  If set, will NOT remove streamlines that deviate from the mean curvature.
	                        Threshold based on --curv_qb_distance.
	
	Pruning options:
	  --min_length MIN_LENGTH
	                        Pruning minimal segment length. [20.0]
	  --max_length MAX_LENGTH
	                        Pruning maximal segment length. [200.0]
	
	Outliers and loops options:
	  --outlier_threshold OUTLIER_THRESHOLD
	                        Outlier removal threshold when using hierarchical QB. [0.5]
	  --loop_max_angle LOOP_MAX_ANGLE
	                        Maximal winding angle over which a streamline is considered as looping. [330.0]
	  --curv_qb_distance CURV_QB_DISTANCE
	                        Clustering threshold for centroids curvature filtering with QB. [10.0]
	
	Saving options:
	  --out_dir OUT_DIR     Output directory for each connection as separate file (.trk).
	  --save_raw_connections
	                        If set, will save all raw cut connections in a subdirectory.
	  --save_intermediate   If set, will save the intermediate results of filtering.
	  --save_discarded      If set, will save discarded streamlines in subdirectories.
	                        Includes loops, outliers and qb_loops.
