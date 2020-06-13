scil_visualize_bundles_mosaic.py
==============

::

	usage: scil_visualize_bundles_mosaic.py [-h] [--uniform_coloring R G B] [--random_coloring SEED]
	                    [--zoom ZOOM] [--ttf TTF] [--ttf_size TTF_SIZE]
	                    [--opacity_background OPACITY_BACKGROUND]
	                    [--resolution_of_thumbnails RESOLUTION_OF_THUMBNAILS] [-f]
	                    in_volume in_bundles [in_bundles ...] out_image
	
	Visualize bundles from a list. The script will output a mosaic (image) with
	screenshots, 6 views per bundle in the list.
	
	positional arguments:
	  in_volume             Volume used as background (e.g. T1, FA, b0).
	  in_bundles            List of tractography files supported by nibabel or binary mask files.
	  out_image             Name of the output image mosaic (e.g. mosaic.jpg, mosaic.png).
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --uniform_coloring R G B
	                        Assign an uniform color to streamlines (or ROIs).
	  --random_coloring SEED
	                        Assign a random color to streamlines (or ROIs).
	  --zoom ZOOM           Rendering zoom. A value greater than 1 is a zoom-in,
	                        a value less than 1 is a zoom-out [1.0].
	  --ttf TTF             Path of the true type font to use for legends.
	  --ttf_size TTF_SIZE   Font size (int) to use for the legends [35].
	  --opacity_background OPACITY_BACKGROUND
	                        Opacity of background image, between 0 and 1.0 [0.4].
	  --resolution_of_thumbnails RESOLUTION_OF_THUMBNAILS
	                        Resolution of thumbnails used in mosaic [300].
	  -f                    Force overwriting of the output files.
