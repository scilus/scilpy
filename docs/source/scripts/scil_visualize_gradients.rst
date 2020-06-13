scil_visualize_gradients.py
==============

::

	usage: scil_visualize_gradients.py [-h] [--dis-sym] [--out_basename OUT_BASENAME] [--res RES]
	                    [--dis-sphere] [--dis-proj] [--plot_shells] [--same-color]
	                    [--opacity OPACITY] [-f] [-v]
	                    gradient_sampling_file [gradient_sampling_file ...]
	
	Vizualisation for gradient sampling.
	Only supports .bvec/.bval and .b (MRtrix).
	
	positional arguments:
	  gradient_sampling_file
	                        Gradient sampling filename. (only accepts .bvec and
	                        .bval together or only .b).
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --dis-sym             Disable antipodal symmetry.
	  --out_basename OUT_BASENAME
	                        Output file name picture without extension (will be
	                        png file(s)).
	  --res RES             Resolution of the output picture(s).
	  -f                    Force overwriting of the output files.
	  -v                    If set, produces verbose output.
	
	Enable/Disable renderings.:
	  --dis-sphere          Disable the rendering of the sphere.
	  --dis-proj            Disable rendering of the projection supershell.
	  --plot_shells         Enable rendering each shell individually.
	
	Rendering options.:
	  --same-color          Use same color for all shell.
	  --opacity OPACITY     Opacity for the shells.
