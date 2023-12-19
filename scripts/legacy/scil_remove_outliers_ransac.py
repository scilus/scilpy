#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scilpy.io.deprecator import deprecate_script
from scripts.scil_volume_remove_outliers_ransac import main as new_main


DEPRECATION_MSG = """
This script has been renamed scil_volume_remove_outliers_ransac.py.
Please change your existing pipelines accordingly.
"""


@deprecate_script("scil_remove_outliers_ransac.py", DEPRECATION_MSG, '1.7.0')
def main():
    new_main()


if __name__ == "__main__":
    main()
