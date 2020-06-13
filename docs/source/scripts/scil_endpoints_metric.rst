scil_endpoints_metric.py
==============

::

	usage: scil_endpoints_metric.py [-h] [--reference REFERENCE] [-f]
	                    in_bundle in_metrics [in_metrics ...] out_folder
	
	Projects metrics onto the endpoints of streamlines. The idea is to visualize
	the cortical areas affected by metrics (assuming streamlines start/end in
	the cortex).
	
	positional arguments:
	  in_bundle             Fiber bundle file.
	  in_metrics            Nifti metric(s) to compute statistics on.
	  out_folder            Folder where to save endpoints metric.
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --reference REFERENCE
	                        Reference anatomy for tck/vtk/fib/dpy file
	                        support (.nii or .nii.gz).
	  -f                    Force overwriting of the output files.
