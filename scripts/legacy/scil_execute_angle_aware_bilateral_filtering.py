#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scilpy.io.deprecator import deprecate_script
from scripts.scil_sh_to_aodf import main as new_main


DEPRECATION_MSG = """
This script has been merged with scil_execute_asymmetric_filtering.py
into scil_sh_to_aodf.py Please change your existing pipelines accordingly.
"""


@deprecate_script("scil_execute_angle_aware_bilateral_filtering.py",
                  DEPRECATION_MSG, '2.0.0')
def main():
    new_main()


if __name__ == "__main__":
    main()
