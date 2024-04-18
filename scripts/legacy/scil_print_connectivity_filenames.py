#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scilpy.io.deprecator import deprecate_script
from scripts.scil_connectivity_print_filenames import main as new_main


DEPRECATION_MSG = """
This script has been renamed scil_connectivity_print_filenames.py.
All our scripts regarding connectivity now start with scil_connectivity_...!

Please change your existing pipelines accordingly.
"""


@deprecate_script("scil_print_connectivity_filenames.py",
                  DEPRECATION_MSG, '2.0.0')
def main():
    new_main()


if __name__ == "__main__":
    main()
