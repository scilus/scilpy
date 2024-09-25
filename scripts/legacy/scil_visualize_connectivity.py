#!/usr/bin/env python
# -*- coding: utf-8 -*-

from scilpy.io.deprecator import deprecate_script
from scripts.scil_viz_connectivity import main as new_main


DEPRECATION_MSG = '''
This script has been renamed scil_viz_connectivity.py. Please change
your existing pipelines accordingly.
'''


@deprecate_script("scil_visualize_connectivity.py", DEPRECATION_MSG, '2.0.0')
def main():
    new_main()


if __name__ == "__main__":
    main()
