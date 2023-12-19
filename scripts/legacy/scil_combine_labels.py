#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scilpy.io.deprecator import deprecate_script
from scripts.scil_labels_combine import main as new_main


DEPRECATION_MSG = """
This script has been renamed scil_labels_combine.py.
Now, all our scripts using labels start with scil_labels_...!

Please change your existing pipelines accordingly.
"""


@deprecate_script("scil_combine_labels.py", DEPRECATION_MSG, '1.7.0')
def main():
    new_main()


if __name__ == "__main__":
    main()
