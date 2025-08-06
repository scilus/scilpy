#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scilpy.io.deprecator import deprecate_script
from scripts.scil_connectivity_graph_measures import main as new_main


DEPRECATION_MSG = """
This script has been renamed scil_connectivity_graph_measures.py.
All our scripts regarding connectivity now start with scil_connectivity_...!

Please change your existing pipelines accordingly.
"""


@deprecate_script("scil_evaluate_connectivity_graph_measures.py",
                  DEPRECATION_MSG, '2.0.0')
def main():
    new_main()


if __name__ == "__main__":
    main()
