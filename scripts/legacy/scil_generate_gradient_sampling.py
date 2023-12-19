#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scilpy.io.deprecator import deprecate_script
from scripts.scil_gradients_generate_sampling import main as new_main


DEPRECATION_MSG = """
This script has been renamed scil_gradients_generate_sampling.py.
Please change your existing pipelines accordingly.
"""


@deprecate_script("scil_generate_gradient_sampling.py", DEPRECATION_MSG,
                  '1.7.0')
def main():
    new_main()


if __name__ == "__main__":
    main()
