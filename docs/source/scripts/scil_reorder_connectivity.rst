scil_reorder_connectivity.py
==============

::

	usage: scil_reorder_connectivity.py [-h] (--in_json IN_JSON | --bct_reorder_nodes OUT_JSON)
	                    [--keys KEYS [KEYS ...]] [--labels_list LABELS_LIST] [-f]
	                    in_matrix out_prefix
	
	Re-order a connectivity matrix using a json file in a format such as:
	    {"temporal": [[1,3,5,7], [0,2,4,6]]}.
	The key is to identify the sub-network, the first list is for the
	column (x) and the second is for the row (y).
	
	The values refers to the coordinates (starting at 0) in the matrix, but if the
	--labels_list parameter is used, the values will refers to the label which will
	be converted to the appropriate coordinates. This file must be the same as the
	one provided to the scil_decompose_connectivity.py
	
	To subsequently use scil_visualize_connectivity.py with a lookup table, you
	must use a label-based reording json and use --labels_list.
	
	The option bct_reorder_nodes creates its own ordering scheme that will be saved
	and then applied to others.
	We recommand running this option on a population-averaged matrix.
	The results are stochastic due to simulated annealing.
	
	This script is under the GNU GPLv3 license, for more detail please refer to
	https://www.gnu.org/licenses/gpl-3.0.en.html
	
	positional arguments:
	  in_matrix             Connectivity matrix in numpy (.npy) format.
	  out_prefix            Prefix for the output matrix filename.
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --in_json IN_JSON     Json file with the sub-network as keys and x/y lists as value.
	  --bct_reorder_nodes OUT_JSON
	                        Rearranges the nodes so the elements are squeezed along the main diagonal.
	  --keys KEYS [KEYS ...]
	                        Only generate the specified sub-network.
	  --labels_list LABELS_LIST
	                        List saved by the decomposition script,
	                        the json must contain labels rather than coordinates.
	  -f                    Force overwriting of the output files.
	
	[1] Rubinov, Mikail, and Olaf Sporns. "Complex network measures of brain
	    connectivity: uses and interpretations." Neuroimage 52.3 (2010):
	    1059-1069.
