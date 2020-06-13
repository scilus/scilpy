scil_print_connectivity_filenames.py
==============

::

	usage: scil_print_connectivity_filenames.py [-h] [-f] in_matrix labels_list out_txt
	
	Output the list of filenames using the coordinates from a binary connectivity
	matrix. Typically used to move around files that are considered valid after
	the scil_filter_connectivity.py script.
	
	Example:
	# Keep connections with more than 1000 streamlines for 100% of a population
	scil_filter_connectivity.py filtering_mask.npy
	    --greater_than */streamlines_count.npy 1000 1.0
	scil_print_connectivity_filenames.py filtering_mask.npy
	    labels_list.txt pass.txt
	for file in $(cat pass.txt);
	    do mv ${SOMEWHERE}/${FILE} ${SOMEWHERE_ELSE}/;
	done
	
	positional arguments:
	  in_matrix    Binary matrix in numpy (.npy) format.
	               Typically from scil_filter_connectivity.py
	  labels_list  List saved by the decomposition script.
	  out_txt      Output text file containing all filenames.
	
	optional arguments:
	  -h, --help   show this help message and exit
	  -f           Force overwriting of the output files.
