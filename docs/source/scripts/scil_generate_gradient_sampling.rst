scil_generate_gradient_sampling.py
==============

::

	usage: scil_generate_gradient_sampling.py [-h] [--eddy] [--duty] [--b0_every B0_EVERY] [--b0_end]
	                    [--b0_value B0_VALUE]
	                    (--bvals bvals [bvals ...] | --b_lin_max B_LIN_MAX | --q_lin_max Q_LIN_MAX)
	                    [--fsl] [--mrtrix] [-v] [-f]
	                    nb_samples [nb_samples ...] out_basename
	
	Generate multi-shell gradient sampling with various processing to accelerate
	acquisition and help artefact correction.
	
	Multi-shell gradient sampling is generated as in [1], the bvecs are then flipped
	to maximize spread for eddy current correction, b0s are interleaved
	at equal spacing and the non-b0 samples are finally shuffled
	to minimize the total diffusion gradient amplitude over a few TR.
	
	positional arguments:
	  nb_samples            Number of samples on each non b0 shell. If multishell,
	                        provide a number per shell.
	  out_basename          Gradient sampling output basename (don't include
	                        extension). Please add options --fsl and/or --mrtrix
	                        below.
	
	Options and Parameters:
	  -h, --help            show this help message and exit
	  --eddy                Apply eddy optimization. B-vectors are flipped to be
	                        well spread without symmetry. [False]
	  --duty                Apply duty cycle optimization. B-vectors are shuffled
	                        to reduce consecutive colinearity in the samples.
	                        [False]
	  --b0_every B0_EVERY   Interleave a b0 every b0_every. No b0 if 0. Only 1 b0
	                        at beginning if > number of samples or negative. [-1]
	  --b0_end              Add a b0 as last sample. [False]
	  --b0_value B0_VALUE   b-value of the b0s. [0.0]
	  --bvals bvals [bvals ...]
	                        bval of each non-b0 shell.
	  --b_lin_max B_LIN_MAX
	                        b-max for linear bval distribution in *b*. [replaces
	                        -bvals]
	  --q_lin_max Q_LIN_MAX
	                        b-max for linear bval distribution in *q*. [replaces
	                        -bvals]
	  -v                    If set, produces verbose output.
	  -f                    Force overwriting of the output files.
	
	Save as:
	  --fsl                 Save in FSL format (.bvec/.bval). [False]
	  --mrtrix              Save in MRtrix format (.b). [False]
	
	References: [1] Emmanuel Caruyer, Christophe Lenglet, Guillermo Sapiro,
	Rachid Deriche. Design of multishell gradient sampling with uniform coverage
	in diffusion MRI. Magnetic Resonance in Medicine, Wiley, 2013, 69 (6),
	pp. 1534-1540. <http://dx.doi.org/10.1002/mrm.24736>
	    
