scil_bundle_mean_std.py
==============

::

	usage: scil_bundle_mean_std.py [-h] [--density_weighting] [--reference REFERENCE]
	                    [--indent INDENT] [--sort_keys]
	                    in_bundle in_metrics [in_metrics ...]
	
	Compute mean and std for the whole bundle for each metric. This is achieved by
	averaging the metrics value of all voxels occupied by the bundle.
	
	Density weighting modifies the contribution of voxel with lower/higher
	streamline count to reduce influence of spurious streamlines.
	
	positional arguments:
	  in_bundle             Fiber bundle file to compute statistics on
	  in_metrics            Nifti file to compute statistics on. Probably some tractometry measure(s) such as FA, MD, RD, ...
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --density_weighting   If set, weight statistics by the number of fibers passing through each voxel.
	  --reference REFERENCE
	                        Reference anatomy for tck/vtk/fib/dpy file
	                        support (.nii or .nii.gz).
	
	Json options:
	  --indent INDENT       Indent for json pretty print.
	  --sort_keys           Sort keys in output json.
