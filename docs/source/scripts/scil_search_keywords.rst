scil_search_keywords.py
==============

::

	usage: scil_search_keywords.py [-h] [--search_parser] [-v] keywords [keywords ...]
	
	Search through all of SCILPY scripts and their docstrings. The output of the
	search will be the intersection of all provided keywords, found either in the
	script name or in its docstring.
	By default, print the matching filenames and the first sentence of the
	docstring. If --verbose if provided, print the full docstring.
	
	Examples:
	    scil_search_keywords.py tractogram filtering
	    scil_search_keywords.py --search_parser tractogram filtering -v
	
	positional arguments:
	  keywords         Search the provided list of keywords.
	
	optional arguments:
	  -h, --help       show this help message and exit
	  --search_parser  Search through and display the full script argparser instead of looking only at the docstring. (warning: much slower).
	  -v               If set, produces verbose output.
