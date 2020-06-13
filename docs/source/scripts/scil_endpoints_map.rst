scil_endpoints_map.py
==============

::

	usage: scil_endpoints_map.py [-h] [--swap] [--indent INDENT] [--sort_keys]
	                    [--reference REFERENCE] [-f]
	                    in_bundle endpoints_map_head endpoints_map_tail
	
	Computes the endpoint map of a bundle. The endpoint map is simply a count of
	the number of streamlines that start or end in each voxel.
	
	The idea is to estimate the cortical area affected by the bundle (assuming
	streamlines start/end in the cortex).
	
	Note: If the streamlines are not ordered the head/tail are random and not
	really two coherent groups. Use the following script to order streamlines:
	scil_uniformize_streamlines_endpoints.py
	
	positional arguments:
	  in_bundle             Fiber bundle filename.
	  endpoints_map_head    Output endpoints map head filename.
	  endpoints_map_tail    Output endpoints map tail filename.
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --swap                Swap head<->tail convention. Can be useful when the reference is not in RAS.
	  --reference REFERENCE
	                        Reference anatomy for tck/vtk/fib/dpy file
	                        support (.nii or .nii.gz).
	  -f                    Force overwriting of the output files.
	
	Json options:
	  --indent INDENT       Indent for json pretty print.
	  --sort_keys           Sort keys in output json.
