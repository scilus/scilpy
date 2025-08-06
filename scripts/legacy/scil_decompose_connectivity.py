#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scilpy.io.deprecator import deprecate_script
from scripts.scil_tractogram_segment_connections_from_labels import main as m


DEPRECATION_MSG = """
This script has been renamed
scil_tractogram_segment_connections_from_labels.py.
Please change your existing pipelines accordingly.
"""


@deprecate_script("scil_decompose_connectivity.py", DEPRECATION_MSG, '2.0.0')
def main():
    m()


if __name__ == "__main__":
    main()
