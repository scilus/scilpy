#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import warnings

from scripts.scil_tractogram_math import main as new_main


def main():
    warnings.simplefilter('always', DeprecationWarning)
    warnings.warn(
        "\n\n*** WARNING ***\n"
        "This script will soon be renamed scil_tractogram_math.py.\n"
        "You should change your existing pipelines accordingly.\n"
        "We will try to keep the following convention: Scripts on "
        "'streamlines' treat each streamline individually.\nScripts on "
        "'tractograms' apply the same operation on all streamlines of the "
        "tractogram.\n\n",
        DeprecationWarning,
        stacklevel=3)

    new_main()


if __name__ == "__main__":
    main()
