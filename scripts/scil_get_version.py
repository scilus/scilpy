#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Search through all of SCILPY scripts and their docstrings. The output of the
search will be the intersection of all provided keywords, found either in the
script name or in its docstring.
By default, print the matching filenames and the first sentence of the
docstring. If --verbose if provided, print the full docstring.

Examples:
    scil_search_keywords.py tractogram filtering
    scil_search_keywords.py --search_parser tractogram filtering -v
"""

import argparse
import ast
import logging
import pathlib
import re
import subprocess

import git
import numpy as np


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    repo_dir = pathlib.Path(__file__).parent.parent
    repo = git.Repo(repo_dir)
    branch = repo.active_branch.name
    origin = repo.remotes.origin.url
    branch = repo.active_branch.name
    last_commit = repo.head.commit
    print(last_commit.message)
    print(last_commit.hexsha)
    print(last_commit.committed_datetime)

    if 'upstream' in repo.git.remote().split():
        upstream = repo.remotes.upstream.url
        upstream_commit = git.cmd.Git().ls_remote(upstream, heads=True)
        # print(upstream_commit)
        count = repo.git.rev_list('--count', 'upstream/master..HEAD')
        print(count)
    # remote_heads = 
    # import time

    # time.asctime(time.gmtime(headcommit.committed_date))
    # time.strftime("%a, %d %b %Y %H:%M", time.gmtime(headcommit.committed_date))
if __name__ == '__main__':
    main()
