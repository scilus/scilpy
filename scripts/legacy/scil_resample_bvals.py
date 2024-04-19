#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scilpy.io.deprecator import deprecate_script
from scripts.scil_gradients_round_bvals import main as new_main


DEPRECATION_MSG = """
This script has been renamed scil_gradients_round_bvals_to_shells.py.
Please change your existing pipelines accordingly. Please note that options
have changed, too.
"""


@deprecate_script("scil_resample_bvals.py", DEPRECATION_MSG, '2.0.0')
def main():
    new_main()


if __name__ == "__main__":
    main()
