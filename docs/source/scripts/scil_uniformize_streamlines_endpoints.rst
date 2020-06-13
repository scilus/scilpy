scil_uniformize_streamlines_endpoints.py
==============

::

	usage: scil_uniformize_streamlines_endpoints.py [-h] (--axis {x,y,z} | --auto) [--swap]
	                    [--reference REFERENCE] [-v] [-f]
	                    in_bundle out_bundle
	
	Uniformize streamlines' endpoints according to a defined axis.
	Useful for tractometry or models creation.
	
	The --auto option will automatically calculate the main orientation.
	If the input bundle is poorly defined, it is possible heuristic will be wrong.
	
	The default is to flip each streamline so their first point's coordinate in the
	defined axis is smaller than their last point (--swap does the opposite).
	
	positional arguments:
	  in_bundle             Input path of the tractography file.
	  out_bundle            Output path of the uniformized file.
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --axis {x,y,z}        Match endpoints of the streamlines along this axis.
	                        SUGGESTION: Commissural = x, Association = y, Projection = z
	  --auto                Match endpoints of the streamlines along an automatically determined axis.
	  --swap                Swap head <-> tail convention. Can be useful when the reference is not in RAS.
	  --reference REFERENCE
	                        Reference anatomy for tck/vtk/fib/dpy file
	                        support (.nii or .nii.gz).
	  -v                    If set, produces verbose output.
	  -f                    Force overwriting of the output files.
