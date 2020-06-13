scil_evaluate_connectivity_measures.py
==============

::

	usage: scil_evaluate_connectivity_measures.py [-h] [--filtering_mask FILTERING_MASK] [--avg_node_wise]
	                    [--append_json] [--small_world] [--indent INDENT]
	                    [--sort_keys] [-v] [-f]
	                    in_conn_matrix in_length_matrix out_json
	
	Evaluate graph theory measures from connectivity matrices.
	A length weighted and a streamline count weighted matrix are required since
	some measures require one or the other.
	
	This script evaluates the measures one subject at the time. To generate a
	population dictionary (similarly to other scil_evaluate_*.py scripts), use the
	--append_json option as well as using the same output filename.
	>>> for i in hcp/*/; do scil_evaluate_connectivity_measures.py ${i}/sc_prob.npy
	    ${i}/len_prob.npy hcp_prob.json --append_json --avg_node_wise; done
	
	Some measures output one value per node, the default behavior is to list
	them all into a list. To obtain only the average use the
	--avg_node_wise option.
	
	The computed connectivity measures are:
	centrality, modularity, assortativity, participation, clustering,
	nodal_strength, local_efficiency, global_efficiency, density, rich_club,
	path_length, edge_count, omega, sigma
	
	For more details about the measures, please refer to
	- https://sites.google.com/site/bctnet/measures
	- https://github.com/aestrivex/bctpy/wiki
	
	This script is under the GNU GPLv3 license, for more detail please refer to
	https://www.gnu.org/licenses/gpl-3.0.en.html
	
	positional arguments:
	  in_conn_matrix        Input connectivity matrix (.npy).
	                        Typically a streamline count weighted matrix.
	  in_length_matrix      Input length weighted matrix (.npy).
	  out_json              Path of the output json.
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --filtering_mask FILTERING_MASK
	                        Binary filtering mask to apply before computing the measures.
	  --avg_node_wise       Return a single value for node-wise measures.
	  --append_json         If the file already exists, will append to the dictionary.
	  --small_world         Compute measure related to small worldness (omega and sigma).
	                         This option is much slower.
	  -v                    If set, produces verbose output.
	  -f                    Force overwriting of the output files.
	
	Json options:
	  --indent INDENT       Indent for json pretty print.
	  --sort_keys           Sort keys in output json.
	
	[1] Rubinov, Mikail, and Olaf Sporns. "Complex network measures of brain
	    connectivity: uses and interpretations." Neuroimage 52.3 (2010):
	    1059-1069.
