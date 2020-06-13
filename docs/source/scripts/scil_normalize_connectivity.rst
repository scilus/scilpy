scil_normalize_connectivity.py
==============

::

	usage: scil_normalize_connectivity.py [-h]
	                    [--length LENGTH_MATRIX | --inverse_length LENGTH_MATRIX]
	                    [--bundle_volume VOLUME_MATRIX]
	                    [--parcel_volume ATLAS LABELS_LIST | --parcel_surface ATLAS LABELS_LIST]
	                    [--max_at_one | --sum_to_one | --log_10] [-f]
	                    in_matrix out_matrix
	
	Normalize a connectivity matrix coming from scil_decompose_connectivity.py.
	3 categories of normalization are available:
	-- Edge attributes
	 - length: Multiply each edge by the average bundle length.
	   Compensate for far away connections when using interface seeding.
	   Cannot be used with inverse_length.
	
	 - inverse_length: Divide each edge by the average bundle length.
	   Compensate for big connections when using white matter seeding.
	   Cannot be used with length.
	
	 - bundle_volume: Divide each edge by the average bundle length.
	   Compensate for big connections when using white matter seeding.
	
	-- Node attributes (Mutually exclusive)
	 - parcel_volume: Divide each edge by the sum of node volume.
	   Compensate for the likelihood of ending in the node.
	   Compensate seeding bias when using interface seeding.
	
	 - parcel_surface: Divide each edge by the sum of the node surface.
	   Compensate for the likelihood of ending in the node.
	   Compensate for seeding bias when using interface seeding.
	
	-- Matrix scaling (Mutually exclusive)
	 - max_at_one: Maximum value of the matrix will be set to one.
	 - sum_to_one: Ensure the sum of all edges weight is one
	 - log_10: Apply a base 10 logarithm to all edges weight
	
	The volume and length matrix should come from the scil_decompose_connectivity.py
	script.
	
	A review of the type of normalization is available in:
	Colon-Perez, Luis M., et al. "Dimensionless, scale-invariant, edge weight
	metric for the study of complex structural networks." PLOS one 10.7 (2015).
	
	However, the proposed weighting of edge presented in this publication is not
	implemented.
	
	positional arguments:
	  in_matrix             Input connectivity matrix. This is typically a streamline_count matrix (.npy).
	  out_matrix            Output normalized matrix (.npy).
	
	optional arguments:
	  -h, --help            show this help message and exit
	  -f                    Force overwriting of the output files.
	
	Edge-wise options:
	  --length LENGTH_MATRIX
	                        Length matrix used for edge-wise multiplication.
	  --inverse_length LENGTH_MATRIX
	                        Length matrix used for edge-wise division.
	  --bundle_volume VOLUME_MATRIX
	                        Volume matrix used for edge-wise division.
	  --parcel_volume ATLAS LABELS_LIST
	                        Atlas and labels list for edge-wise division.
	  --parcel_surface ATLAS LABELS_LIST
	                        Atlas and labels list for edge-wise division.
	
	Scaling options:
	  --max_at_one          Scale matrix with maximum value at one.
	  --sum_to_one          Scale matrix with sum of all elements at one.
	  --log_10              Apply a base 10 logarithm to the matrix.
