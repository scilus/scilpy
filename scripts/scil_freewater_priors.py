#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute the axial (para_diff) and mean (iso_diff) diffusivity priors for
freewater.
"""

from scripts.scil_NODDI_priors import main as noddi_priors_main

EPILOG = """
Reference:
    [1] Zhang H, Schneider T, Wheeler-Kingshott CA, Alexander DC.
        NODDI: practical in vivo neurite orientation dispersion
        and density imaging of the human brain.
        NeuroImage. 2012 Jul 16;61:1000-16.
"""


def main():
    noddi_priors_main()


if __name__ == "__main__":
    main()
