#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import warnings

from scilpy.io.deprecator import deprecate_script
from scripts.scil_tractogram_math import main as new_main


DEPRECATION_MSG = """

*** WARNING ***

This script will soon be renamed scil_tractogram_math.py. You should change
your existing pipelines accordingly. We will try to keep the following
convention:

- Scripts on 'streamlines' treat each streamline individually.
- Scripts on 'tractograms' apply the same operation on all streamlines of the
  tractogram.
"""

@deprecate_script(DEPRECATION_MSG, '1.5.0')
def main():
    new_main()


if __name__ == "__main__":
    main()
