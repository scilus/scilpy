scil_bundle_mean_std_per_point.py
==============

::

	usage: scil_bundle_mean_std_per_point.py [-h] [--density_weighting] [--distance_weighting]
	                    [--out_json OUT_JSON] [-f] [--reference REFERENCE]
	                    [--indent INDENT] [--sort_keys]
	                    in_bundle label_map distance_map metrics [metrics ...]
	
	Compute mean and standard deviation for all streamlines points in the bundle
	for each metric combination, along the bundle, i.e. for each point.
	
	**To create label_map and distance_map, see scil_label_and_distance_maps.py.
	
	positional arguments:
	  in_bundle             Fiber bundle file to compute statistics on.
	  label_map             Label map (.npz) of the corresponding fiber bundle.
	  distance_map          Distance map (.npz) of the corresponding bundle/centroid streamline.
	  metrics               Nifti file to compute statistics on. Probably some tractometry measure(s) such as FA, MD, RD, ...
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --density_weighting   If set, weight statistics by the number of streamlines passing through each voxel.
	  --distance_weighting  If set, weight statistics by the inverse of the distance between a streamline and the centroid.
	  --out_json OUT_JSON   Path of the output json file. If not given, json formatted stats are simply printed.
	  -f                    Force overwriting of the output files.
	  --reference REFERENCE
	                        Reference anatomy for tck/vtk/fib/dpy file
	                        support (.nii or .nii.gz).
	
	Json options:
	  --indent INDENT       Indent for json pretty print.
	  --sort_keys           Sort keys in output json.
