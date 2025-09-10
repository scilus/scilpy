#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download data for tests
"""

import tqdm

import nltk

from scilpy.io.dvc import pull_test_case_package
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

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


def main():

    tqdm_bar = tqdm.tqdm(total=len(LIST_ZIP_FILES)+2,
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
