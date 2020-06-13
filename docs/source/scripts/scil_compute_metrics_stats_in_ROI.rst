scil_compute_metrics_stats_in_ROI.py
==============

::

	usage: scil_compute_metrics_stats_in_ROI.py [-h]
	                    (--metrics_dir METRICS_DIR | --metrics METRICS_FILE_LIST [METRICS_FILE_LIST ...])
	                    [--bin] [--normalize_weights] [-f] [--indent INDENT]
	                    [--sort_keys]
	                    in_mask
	
	Compute the statistics (mean, std) of scalar maps, which can represent
	diffusion metrics, in a ROI.
	
	The mask can either be a binary mask, or a weighting mask. If the mask is
	a weighting mask it should either contain floats between 0 and 1 or should be
	normalized with --normalize_weights.
	
	IMPORTANT: if the mask contains weights (and not 0 and 1 exclusively), the
	standard deviation will also be weighted.
	
	positional arguments:
	  in_mask               Mask volume filename.
	                        Can be a binary mask or a weighted mask.
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --bin                 If set, will consider every value of the mask higher than 0 to be part of the mask, and set to 1 (equivalent weighting for every voxel).
	  --normalize_weights   If set, the weights will be normalized to the [0,1] range.
	  -f                    Force overwriting of the output files.
	
	Metrics input options:
	  --metrics_dir METRICS_DIR
	                        Metrics files directory. Name of the directory containing the metrics files.
	  --metrics METRICS_FILE_LIST [METRICS_FILE_LIST ...]
	                        Metrics nifti filename. List of the names of the metrics file, in nifti format.
	
	Json options:
	  --indent INDENT       Indent for json pretty print.
	  --sort_keys           Sort keys in output json.
