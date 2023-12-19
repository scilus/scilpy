#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scilpy.io.deprecator import deprecate_script
from scripts.scil_gradients_modify_axes import main as new_main


DEPRECATION_MSG = """
This script has been renamed scil_gradients_modify_axes.py.
Please change your existing pipelines accordingly. Please note that options
have changed, too.
"""


@deprecate_script("scil_flip_gradients.py", DEPRECATION_MSG, '1.7.0')
def main():
    new_main()


if __name__ == "__main__":
    main()
