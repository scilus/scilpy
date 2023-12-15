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
import platform
import os
import time
from scilpy.io.utils import add_verbose_arg

def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('--show_dependencies', action='store_true',
                   help='Show the dependencies of scilpy.')
    add_verbose_arg(p)
    return p


def _bold(string):
    string = str(string)
    return '\033[1m' + string + '\033[0m'


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    print('Your {} version is: {}'.format(_bold('Python'),
                                          _bold(platform.python_version())))

    dists = [d for d in pkg_resources.working_set]
    important_deps = ['dipy', 'numpy', 'nibabel', 'dmri-commit', 'dmri-amico']

    for dist in dists:
        if dist.project_name == 'scilpy':
            date = time.ctime(os.path.getctime(dist.location))
            date = datetime.datetime.strptime(date, "%a %b %d %H:%M:%S %Y")
            date = date.strftime('%Y-%m-%d')
            print('You installed {} with pip on {}'.format(_bold('Scilpy'),
                                                           _bold(date)))
            print('The closest release is {}\n'.format(_bold(dist.version)))
        if args.show_dependencies and dist.project_name in important_deps:
            print('     {}: {}'.format(_bold(dist.project_name),
                  _bold(dist.version)))

    repo_dir = pathlib.Path(__file__).parent.parent
    try:
        repo = git.Repo(repo_dir)
    except git.InvalidGitRepositoryError:
        print('Your Scilpy directory is: {}'.format(_bold(repo_dir)))
        print('It is not a git repository, so we cannot give you more '
              'information.')
        return
    if not repo.head.is_detached:
        branch = repo.active_branch
        origin = repo.remotes.origin.url
    else:
        with open(os.path.join(repo_dir, '.git', 'FETCH_HEAD')) as f:
            git_text = f.read().split()
            branch = git_text[2].replace("'", "")
            origin = git_text[4]

    print('Your Scilpy directory is: {}'.format(_bold(repo_dir)))
    print('Your current Origin is: {}'.format(_bold(origin)))
    print('Your repository is on branch: {}\n'.format(_bold(branch)))

    last_commit = repo.head.commit
    print('The last commit hash is: {}'.format(_bold(last_commit.hexsha)))
    date = str(last_commit.committed_datetime).split()[0]
    print('The last commit date is: {}, by {}'.format(_bold(date),
          _bold(last_commit.author)))
    print('The last commit message is: {}'.format(_bold(last_commit.message)))

    upstream_url = ['git@github.com:scilus/scilpy.git',
                    'https://github.com/scilus/scilpy.git']
    origin = repo.remotes.origin.url
    if 'upstream' in repo.git.remote().split() or \
            origin in upstream_url:

        if origin in upstream_url:
            upstream = origin
            count = repo.git.rev_list('--count', 'origin/master..HEAD',
                                      '--left-right').split()
        else:
            upstream = repo.remotes.upstream.url
            count = repo.git.rev_list('--count', 'upstream/master..HEAD',
                                      '--left-right').split()

        remote_name = 'origin' if origin in upstream_url else 'upstream'
        print('Your upstream is set to: {}'.format(_bold(upstream)))
        print('You are {} commits behind {}/master'.format(
            _bold(count[0]), remote_name))
        print('You are {} commits ahead of {}/master'.format(
            _bold(count[1]), remote_name))


if __name__ == '__main__':
    main()
