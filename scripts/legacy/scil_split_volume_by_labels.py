#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scilpy.io.deprecator import deprecate_script
from scripts.scil_labels_split_volume_from_lut import main as new_main


DEPRECATION_MSG = """
This script has been renamed scil_labels_split_volume_from_lut.
Now, all our scripts using labels start with scil_labels_...!

Please change your existing pipelines accordingly.
"""


@deprecate_script("scil_split_volume_by_labels.py", DEPRECATION_MSG, '2.0.0')
def main():
    new_main()


if __name__ == "__main__":
    main()
