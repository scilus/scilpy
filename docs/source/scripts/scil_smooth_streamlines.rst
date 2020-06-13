scil_smooth_streamlines.py
==============

::

	usage: scil_smooth_streamlines.py [-h] (--gaussian SIGMA | --spline SIGMA NB_CTRL_POINT)
	                    [-e ERROR_RATE] [--reference REFERENCE] [-f] [-v]
	                    in_tractogram out_tractogram
	
	This script will smooth the streamlines, usually to remove the
	'wiggles' in probabilistic tracking.
	Two choices of methods are available:
	- Gaussian will use the surrounding coordinates for smoothing.
	Streamlines are resampled to 1mm step-size and the smoothing is
	performed on the coordinate array. The sigma will be indicative of the
	number of points surrounding the center points to be used for blurring.
	
	- Spline will fit a spline curve to every streamline using a sigma and
	the number of control points. The sigma represents the allowed distance
	from the control points. The control points for the spline fit will be
	the resampled streamline.
	
	This script enforces endpoints to remain the same.
	
	WARNING:
	- too low of a sigma (e.g: 1) with a lot of control points (e.g: 15)
	will create crazy streamlines that could end up out of the bounding box.
	- data_per_point will be lost.
	
	positional arguments:
	  in_tractogram         Input tractography file.
	  out_tractogram        Output tractography file.
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --gaussian SIGMA      Sigma for smoothing. Use the value of surronding
	                        X,Y,Z points on the streamline to blur the streamlines.
	                        A good sigma choice would be around 5.
	  --spline SIGMA NB_CTRL_POINT
	                        Sigma for smoothing. Model each streamline as a spline.
	                        A good sigma choice would be around 5 and control point around 10.
	  -e ERROR_RATE         Maximum compression distance in mm after smoothing. [0.1]
	  --reference REFERENCE
	                        Reference anatomy for tck/vtk/fib/dpy file
	                        support (.nii or .nii.gz).
	  -f                    Force overwriting of the output files.
	  -v                    If set, produces verbose output.
