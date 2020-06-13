scil_visualize_connectivity.py
==============

::

	usage: scil_visualize_connectivity.py [-h] [--labels_list LABELS_LIST] [--reorder_json FILE KEY]
	                    [--lookup_table LOOKUP_TABLE] [--name_axis]
	                    [--axis_text_size X_SIZE Y_SIZE]
	                    [--axis_text_angle X_ANGLE Y_ANGLE] [--colormap COLORMAP]
	                    [--display_legend] [--write_values FONT_SIZE DECIMAL]
	                    [--histogram FILENAME] [--nb_bins NB_BINS]
	                    [--exclude_zeros] [--log] [--show_only] [-f]
	                    in_matrix out_png
	
	Script to display a connectivity matrix and adjust the desired visualization.
	Made to work with scil_decompose_connectivity.py and
	scil_reorder_connectivity.py.
	
	This script can either display the axis labels as:
	- Coordinates (0..N)
	- Labels (using --labels_list)
	- Names (using --labels_list and --lookup_table)
	
	If the matrix was made from a bigger matrix using scil_reorder_connectivity.py,
	provide the json and specify the key (using --reorder_json)
	
	positional arguments:
	  in_matrix             Connectivity matrix in numpy (.npy) format.
	  out_png               Output filename for the figure.
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --log                 Apply a base 10 logarithm to the matrix.
	  --show_only           Do not save the figure, simply display it.
	  -f                    Force overwriting of the output files.
	
	Naming options:
	  --labels_list LABELS_LIST
	                        List saved by the decomposition script,
	                        the json must contain labels rather than coordinates.
	  --reorder_json FILE KEY
	                        Json file with the sub-network as keys and x/y lists as value AND the key to use.
	  --lookup_table LOOKUP_TABLE
	                        Lookup table with the label number as keys and the name as values.
	
	Matplotlib options:
	  --name_axis           Use the provided info/files to name axis.
	  --axis_text_size X_SIZE Y_SIZE
	                        Font size of the X and Y axis labels. [(10, 10)]
	  --axis_text_angle X_ANGLE Y_ANGLE
	                        Text angle of the X and Y axis labels. [(90, 0)]
	  --colormap COLORMAP   Colormap to use for the matrix. [viridis]
	  --display_legend      Display the colorbar next to the matrix.
	  --write_values FONT_SIZE DECIMAL
	                        Write the values at the center of each node.
	                        The font size and the rouding parameters can be adjusted.
	
	Histogram options:
	  --histogram FILENAME  Compute and display/save an histogram of weigth.
	  --nb_bins NB_BINS     Number of bins to use for the histogram.
	  --exclude_zeros       Exclude the zeros from the histogram.
