#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scilpy.io.deprecator import deprecate_script
from scripts.scil_bundle_print_shape_measures import main as new_main


DEPRECATION_MSG = """
This script has been deleted, but you can now obtain the same information
from scil_bundle_print_shape_measures.py.

Please change your existing pipelines accordingly.
"""


@deprecate_script("scil_compute_bundle_volume.py", DEPRECATION_MSG, '1.7.0')
def main():
    new_main()


if __name__ == "__main__":
    main()
