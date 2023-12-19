#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scilpy.io.deprecator import deprecate_script
from scripts.scil_tractogram_convert_hdf5_to_trk import main as new_main


DEPRECATION_MSG = """
This script has been renamed scil_tractogram_convert_hdf5_to_trk.py.
Please change your existing pipelines accordingly.
"""


@deprecate_script("scil_save_connections_from_hdf5.py", DEPRECATION_MSG,
                  '1.7.0')
def main():
    new_main()


if __name__ == "__main__":
    main()
