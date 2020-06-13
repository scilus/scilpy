scil_filter_connectivity.py
==============

::

	usage: scil_filter_connectivity.py [-h] [--lower_than [LOWER_THAN [LOWER_THAN ...]]]
	                    [--greater_than [GREATER_THAN [GREATER_THAN ...]]]
	                    [--keep_condition_count] [--inverse_mask] [-v] [-f]
	                    out_matrix_mask
	
	Script to facilitate filtering of connectivity matrices.
	The same could be achieved through a complex sequence of scil_connectivity_math.py.
	
	Can be used with any connectivity matrix from scil_compute_connectivity.py.
	
	For example, a simple filtering (Jasmeen style) would be:
	scil_filter_connectivity.py out_mask.npy
	    --greater_than */sc.npy 1 0.90
	    --lower_than */sim.npy 2 0.90
	    --greater_than */len.npy 40 0.90 -v;
	
	This will result in a binary mask where each node with a value of 1 represents
	a node with at least 90% of the population having at least 1 streamline,
	90% of the population is similar to the average (2mm) and 90% of the
	population having at least 40mm of average streamlines length.
	
	--greater_than or --lower_than expect the same convention:
	    MATRICES_LIST VALUE_THR POPULATION_PERC
	It is strongly recommended (but not enforced) that the same number of
	connectivity matrices is used for each condition.
	
	This script performs an intersection of all conditions, meaning that all
	conditions must be met in order not to be filtered.
	If the user wants to manually handle the requirements, --keep_condition_count
	can be used and manually binarized using scil_connectivity_math.py
	
	positional arguments:
	  out_matrix_mask       Output mask (matrix) resulting from the provided conditions (.npy).
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --lower_than [LOWER_THAN [LOWER_THAN ...]]
	                        Lower than condition using the VALUE_THR in at least POPULATION_PERC (from MATRICES_LIST).
	                        See description for more details.
	  --greater_than [GREATER_THAN [GREATER_THAN ...]]
	                        Greater than condition using the VALUE_THR in at least POPULATION_PERC (from MATRICES_LIST).
	                        See description for more details.
	  --keep_condition_count
	                        Report the number of condition(s) that pass/fail rather than a binary mask.
	  --inverse_mask        Inverse the final mask. 0 where all conditions are respected and 1 where at least one fail.
	  -v                    If set, produces verbose output.
	  -f                    Force overwriting of the output files.
