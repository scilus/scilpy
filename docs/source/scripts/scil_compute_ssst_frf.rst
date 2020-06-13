scil_compute_ssst_frf.py
==============

::

	usage: scil_compute_ssst_frf.py [-h] [--force_b0_threshold] [--mask MASK] [--mask_wm]
	                    [--fa FA_THRESH] [--min_fa MIN_FA_THRESH]
	                    [--min_nvox MIN_NVOX] [--roi_radius ROI_RADIUS]
	                    [--roi_center tuple3] [-f] [-v]
	                    input bvals bvecs frf_file
	
	Compute a single Fiber Response Function from a DWI.
	
	A DTI fit is made, and voxels containing a single fiber population are
	found using a threshold on the FA.
	
	positional arguments:
	  input                 Path of the input diffusion volume.
	  bvals                 Path of the bvals file, in FSL format.
	  bvecs                 Path of the bvecs file, in FSL format.
	  frf_file              Path to the output FRF file, in .txt format, saved by
	                        Numpy.
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --force_b0_threshold  If set, the script will continue even if the minimum
	                        bvalue is suspiciously high ( > 20)
	  --mask MASK           Path to a binary mask. Only the data inside the mask
	                        will be used for computations and reconstruction.
	                        Useful if no white matter mask is available.
	  --mask_wm             Path to a binary white matter mask. Only the data
	                        inside this mask and above the threshold defined by
	                        --fa will be used to estimate the fiber response
	                        function.
	  --fa FA_THRESH        If supplied, use this threshold as the initial
	                        threshold to select single fiber voxels. [0.7]
	  --min_fa MIN_FA_THRESH
	                        If supplied, this is the minimal value that will be
	                        tried when looking for single fiber voxels. [0.5]
	  --min_nvox MIN_NVOX   Minimal number of voxels needing to be identified as
	                        single fiber voxels in the automatic estimation. [300]
	  --roi_radius ROI_RADIUS
	                        If supplied, use this radius to select single fibers
	                        from the tensor to estimate the FRF. The roi will be a
	                        cube spanning from the middle of the volume in each
	                        direction. [10]
	  --roi_center tuple(3)
	                        If supplied, use this center to span the roi of size
	                        roi_radius. [center of the 3D volume]
	  -f                    Force overwriting of the output files.
	  -v                    If set, produces verbose output.
	
	References: [1] Tournier et al. NeuroImage 2007
