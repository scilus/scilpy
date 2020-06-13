scil_plot_mean_std_per_point.py
==============

::

	usage: scil_plot_mean_std_per_point.py [-h] [--fill_color FILL_COLOR] [-f] in_json out_dir
	
	Plot mean/std per point.
	
	positional arguments:
	  in_json               JSON file containing the mean/std per point. For example, can be created using scil_compute_metrics_along_streamline.
	  out_dir               Output directory.
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --fill_color FILL_COLOR
	                        Hexadecimal RGB color filling the region between mean ± std. The hexadecimal RGB color should be formatted as 0xRRGGBB.
	  -f                    Force overwriting of the output files.
