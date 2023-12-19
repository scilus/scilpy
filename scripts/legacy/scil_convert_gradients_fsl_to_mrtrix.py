#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scilpy.io.deprecator import deprecate_script
from scripts.scil_gradients_convert_fsl_to_mrtrix import main as new_main


DEPRECATION_MSG = """
This script has been renamed scil_gradients_convert_fsl_to_mrtrix.py.
Please change your existing pipelines accordingly.
"""


@deprecate_script("scil_convert_gradients_fsl_to_mrtrix.py",
                  DEPRECATION_MSG, '1.7.0')
def main():
    new_main()


if __name__ == "__main__":
    main()
