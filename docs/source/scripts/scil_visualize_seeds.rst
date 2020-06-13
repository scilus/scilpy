scil_visualize_seeds.py
==============

::

	usage: scil_visualize_seeds.py [-h] [--save SAVE] [-f] tractogram
	
	Visualize seeds used to generate the tractogram or bundle.
	When tractography was run, each streamline produced by the tracking algorithm
	saved its seeding point (its origin).
	
	The tractogram must have been generated from scil_compute_local/pft_tracking.py
	with the --save_seeds option.
	
	positional arguments:
	  tractogram   Tractogram file (must be trk)
	
	optional arguments:
	  -h, --help   show this help message and exit
	  --save SAVE  If set, save a screenshot of the result in the specified filename
	  -f           Force overwriting of the output files.
