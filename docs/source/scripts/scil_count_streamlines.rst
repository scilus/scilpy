scil_count_streamlines.py
==============

::

	usage: scil_count_streamlines.py [-h] [--indent INDENT] [--sort_keys] in_tractogram
	
	Return the number of streamlines in a tractogram. Only support trk and tck in
	order to support the lazy loading from nibabel.
	
	positional arguments:
	  in_tractogram    Path of the input tractogram file.
	
	optional arguments:
	  -h, --help       show this help message and exit
	
	Json options:
	  --indent INDENT  Indent for json pretty print.
	  --sort_keys      Sort keys in output json.
