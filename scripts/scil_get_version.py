#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Give you information about your current scilpy installation.
This is useful for non-developers to give you the information
needed to reproduce your results, or to help debugging.

If you are experiencing a bug, please run this script and
send the output to the scilpy developers.
"""

import argparse
import datetime
import git
import pathlib
import pkg_resources
import os
import time


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    return p


def _bold(string):
    string = str(string)
    return '\033[1m' + string + '\033[0m'


def main():
    parser = _build_arg_parser()

    dists = [d for d in pkg_resources.working_set]
    for dist in dists:
        if dist.project_name == 'scilpy':
            date = time.ctime(os.path.getctime(dist.location))
            date = datetime.datetime.strptime(date, "%a %b %d %H:%M:%S %Y")
            date = date.strftime('%Y-%m-%d')
            print('You installed Scilpy with pip on {}'.format(_bold(date)))
            print('The closest release is {}\n'.format(_bold(dist.version)))

    repo_dir = pathlib.Path(__file__).parent.parent
    repo = git.Repo(repo_dir)
    branch = repo.active_branch.name
    origin = repo.remotes.origin.url
    branch = repo.active_branch.name
    print('Your Scilpy directory is: {}'.format(_bold(repo_dir)))
    print('Your current Origin is: {}'.format(_bold(origin)))
    print('Your repository is on branch: {}\n'.format(_bold(branch)))

    last_commit = repo.head.commit
    print('The last commit hash is: {}'.format(_bold(last_commit.hexsha)))
    print('The last commit author is: {}'.format(_bold(last_commit.author)))
    date = str(last_commit.committed_datetime).split()[0]
    print('The last commit date is: {}, by {}'.format(_bold(date),
          _bold(last_commit.author)))
    print('The last commit message is: {}'.format(_bold(last_commit.message)))

    upstream_url = ['git@github.com:scilus/scilpy.git',
                    'https://github.com/scilus/scilpy.git']
    if 'upstream' in repo.git.remote().split() or \
            origin in upstream_url:

        if origin in upstream_url:
            upstream = origin
        else:
            upstream = repo.remotes.upstream.url

        last_commit = git.cmd.Git().ls_remote(upstream, heads=True).split()[0]
        count = repo.git.rev_list('--count', 'upstream/master..HEAD',
                                  '--left-right').split()

        print('Your upstream is set to: {}'.format(_bold(upstream)))
        print('The last commit hash on upstream is: {}'.format(
            _bold(last_commit)))

        print('You are {} commits behind upstream/master'.format(

            _bold(count[0])))
        print('You are {} commits ahead of upstream/master'.format(
            _bold(count[1])))


if __name__ == '__main__':
    main()
