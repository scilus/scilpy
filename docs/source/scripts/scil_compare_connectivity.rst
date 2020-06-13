scil_compare_connectivity.py
==============

::

	usage: scil_compare_connectivity.py [-h] --in_g1 IN_G1 [IN_G1 ...] --in_g2 IN_G2 [IN_G2 ...]
	                    [--tail {left,right,both}] [--paired]
	                    [--fdr | --bonferroni] [--p_threshold THRESH OUT_FILE]
	                    [--filtering_mask FILTERING_MASK] [--processes NBR] [-v]
	                    [-f]
	                    out_pval_matrix
	
	Performs a network-based statistical comparison for populations g1 and g2. The
	output is a matrix of the same size as the input connectivity matrices, with
	p-values at each edge.
	All input matrices must have the same shape (NxN). For paired t-test, both
	groups must have the same number of observations.
	
	For example, if you have streamline count weighted matrices for a MCI and a
	control group and you want to investiguate differences in their connectomes:
	>>> scil_compare_connectivity.py pval.npy --g1 MCI/*_sc.npy --g2 CTL/*_sc.npy
	
	--filtering_mask will simply multiply the binary mask to all input
	matrices before performing the statistical comparison. Reduces the number of
	statistical tests, useful when using --fdr or --bonferroni.
	
	positional arguments:
	  out_pval_matrix       Output matrix (.npy) containing the edges p-value.
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --in_g1 IN_G1 [IN_G1 ...]
	                        List of matrices for the first population (.npy).
	  --in_g2 IN_G2 [IN_G2 ...]
	                        List of matrices for the second population (.npy).
	  --tail {left,right,both}
	                        Enables specification of an alternative hypothesis:
	                        left: mean of g1 < mean of g2,
	                        right: mean of g2 < mean of g1,
	                        both: both means are not equal (default).
	  --paired              Use paired sample t-test instead of population t-test.
	                        --in_g1 and --in_g2 must be ordered the same way.
	  --fdr                 Perform a false discovery rate (FDR) correction for the p-values.
	                        Uses the number of non-zero edges as number of tests (value between 0.01 and 0.1).
	  --bonferroni          Perform a Bonferroni correction for the p-values.
	                        Uses the number of non-zero edges as number of tests.
	  --p_threshold THRESH OUT_FILE
	                        Threshold the final p-value matrix and save the binary matrix (.npy).
	  --filtering_mask FILTERING_MASK
	                        Binary filtering mask (.npy) to apply before computing the measures.
	  --processes NBR       Number of sub-processes to start. 
	                        Default: [1]
	  -v                    If set, produces verbose output.
	  -f                    Force overwriting of the output files.
	
	[1] Rubinov, Mikail, and Olaf Sporns. "Complex network measures of brain
	    connectivity: uses and interpretations." Neuroimage 52.3 (2010):
	    1059-1069.
	[2] Zalesky, Andrew, Alex Fornito, and Edward T. Bullmore. "Network-based
	    statistic: identifying differences in brain networks." Neuroimage 53.4
	    (2010): 1197-1207.
