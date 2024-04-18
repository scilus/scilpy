#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scilpy.io.deprecator import deprecate_script
from scripts.scil_json_merge_entries import main as new_main


DEPRECATION_MSG = """
This script has been renamed scil_json_merge_entries.py.
Please change your existing pipelines accordingly.
"""


@deprecate_script("scil_json_merge_entries.py",
                  DEPRECATION_MSG, '2.0.0')
def main():
    new_main()


if __name__ == "__main__":
    main()
