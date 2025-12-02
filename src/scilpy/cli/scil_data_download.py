#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download data for tests.
This is used to download our internal tests into your home/.scilpy.
"""
import argparse
import logging

import tqdm

import nltk

from scilpy.io.dvc import pull_test_case_package
from scilpy.io.fetcher import fetch_data, get_testing_files_dict
from scilpy.io.utils import add_verbose_arg
from scilpy.version import version_string

LIST_ZIP_FILES = ["anatomical_filtering",
                  "atlas",
                  "bids_json",
                  "bst",
                  "btensor_testdata",
                  "bundles",
                  "commit_amico",
                  "connectivity",
                  "filtering",
                  "ihMT",
                  "lesions",
                  "mrds",
                  "MT",
                  "others",
                  "plot",
                  "processing",
                  "stats",
                  "surface_vtk_fib",
                  "tracking",
                  "tractograms",
                  "tractometry"]

def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)
    add_verbose_arg(p)
    return p

def main():

    # No argument but adding the arg parser in case user wants to do --help.
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    tqdm_bar = tqdm.tqdm(total=len(LIST_ZIP_FILES)+1,
                         desc="Download data test")
    for zip_file in LIST_ZIP_FILES:
        fetch_data(get_testing_files_dict(),
                   keys=[zip_file+'.zip'], verbose=False)
        tqdm_bar.update(1)

    _ = pull_test_case_package("aodf")
    tqdm_bar.update(1)

    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('wordnet', quiet=True)


if __name__ == "__main__":
    main()
