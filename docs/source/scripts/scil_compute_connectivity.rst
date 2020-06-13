scil_compute_connectivity.py
==============

::

	usage: scil_compute_connectivity.py [-h] [--volume OUT_FILE] [--streamline_count OUT_FILE]
	                    [--length OUT_FILE] [--similarity IN_FOLDER OUT_FILE]
	                    [--maps IN_FOLDER OUT_FILE] [--metrics IN_FILE OUT_FILE]
	                    [--density_weighting] [--no_self_connection]
	                    [--include_dps] [--force_labels_list FORCE_LABELS_LIST]
	                    [--processes NBR] [-v] [-f]
	                    in_hdf5 in_labels
	
	This script computes a variety of measures in the form of connectivity
	matrices. This script is made to follow scil_decompose_connectivity and
	uses the same labels list as input.
	
	The script expects a folder containing all relevants bundles following the
	naming convention LABEL1_LABEL2.trk and a text file containing the list of
	labels that should be part of the matrices. The ordering of labels in the
	matrices will follow the same order as the list.
	This script only generates matrices in the form of array, does not visualize
	or reorder the labels (node).
	
	The parameter --similarity expects a folder with density maps (LABEL1_LABEL2.nii.gz)
	following the same naming convention as the input directory.
	The bundles should be averaged version in the same space. This will
	compute the weighted-dice between each node and their homologuous average
	version.
	
	The parameters --metrics can be used more than once and expect a map (t1, fa,
	etc.) in the same space and each will generate a matrix. The average value in
	the volume occupied by the bundle will be reported in the matrices nodes.
	
	The parameters --maps can be used more than once and expect a folder with
	pre-computed maps (LABEL1_LABEL2.nii.gz) following the same naming convention
	as the input directory. Each will generate a matrix. The average non-zeros
	value in the map will be reported in the matrices nodes.
	
	positional arguments:
	  in_hdf5               Input filename for the hdf5 container (.h5).
	                        Obtained from scil_decompose_connectivity.py.
	  in_labels             Labels file name (nifti).
	                        This generates a NxN connectivity matrix.
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --volume OUT_FILE     Output file for the volume weighted matrix (.npy).
	  --streamline_count OUT_FILE
	                        Output file for the streamline count weighted matrix (.npy).
	  --length OUT_FILE     Output file for the length weighted matrix (.npy).
	  --similarity IN_FOLDER OUT_FILE
	                        Input folder containing the averaged bundle density
	                        maps (.nii.gz) and output file for the similarity weighted matrix (.npy).
	  --maps IN_FOLDER OUT_FILE
	                        Input folder containing pre-computed maps (.nii.gz)
	                        and output file for the weighted matrix (.npy).
	  --metrics IN_FILE OUT_FILE
	                        Input (.nii.gz). and output file (.npy) for a metric weighted matrix.
	  --density_weighting   Use density-weighting for the metric weighted matrix.
	  --no_self_connection  Eliminate the diagonal from the matrices.
	  --include_dps         Save matrices from data_per_streamline.
	  --force_labels_list FORCE_LABELS_LIST
	                        Path to a labels list (.txt) in case of missing labels in the atlas.
	  --processes NBR       Number of sub-processes to start. 
	                        Default: [1]
	  -v                    If set, produces verbose output.
	  -f                    Force overwriting of the output files.
