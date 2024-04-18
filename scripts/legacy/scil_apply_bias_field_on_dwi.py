#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scilpy.io.deprecator import deprecate_script
from scripts.scil_dwi_apply_bias_field import main as new_main


DEPRECATION_MSG = """
This script has been renamed scil_dwi_apply_bias_field.py.
Please change your existing pipelines accordingly.
"""


@deprecate_script("scil_apply_bias_field_on_dwi.py", DEPRECATION_MSG, '2.0.0')
def main():
    new_main()


if __name__ == "__main__":
    main()
