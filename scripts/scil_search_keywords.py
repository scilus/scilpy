#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Search through all of SCILPY scripts and their docstrings. The output of the
search will be the intersection of all provided keywords, found either in the
script name or in its docstring.
"""

import argparse
import ast
import inspect
import os
import pathlib
import re

import numpy as np
import scilpy

RED = '\033[31m'
BOLD = '\033[1m'
ENDC_1 = '\033[0m'
ENDC_2 = '\033[37m'
spacing = '=================='


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('keywords', nargs='+',
                   help='Search the provided keywords.')

    p.add_argument('--show_docstring', action='store_true',
                   help='Display the script full docstring.')
    p.add_argument('--show_help', action='store_true',
                   help='Display the script full argparser (much slower).')
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Use directory of this script, should work with most installation setups
    script_dir = pathlib.Path(__file__).parent
    matches = []

    kw_subs = [re.compile('(' + re.escape(kw) + ')', re.IGNORECASE)
               for kw in args.keywords]

    counter = 0
    for script in sorted(script_dir.glob('*.py')):
        filename = script.name

        if args.show_help:
            # Display full docstring
            help_print = os.popen('{} --help'.format(script))
            display_info = help_print.read()
        else:
            # Extract first sentence by finding the first dot
            docstring = _get_docstring(str(script))
            if args.show_docstring:
                display_info = docstring
            else:
                display_info = _extract_first_sentence(docstring)
                display_info = display_info.replace('\n', ' ')

        # Test intersection of all keywords, either in filename or docstring
        if not _test_matching_keywords(args.keywords, [filename, display_info]):
            continue

        matches.append(filename)
        display_info = display_info or 'No docstring available!'

        # new_key calls regex group \1 to keep the same case
        # Alternate light gray and white for easier reading
        if counter % 2 == 0:
            color_scheme = ENDC_1
        else:
            color_scheme = ENDC_2

        display_info = color_scheme + display_info
        new_key = '{}\\1{}'.format(RED + BOLD, ENDC_1+color_scheme)

        for regex in kw_subs:
            filename = regex.sub(new_key, filename)
            display_info = regex.sub(new_key, display_info)

        if args.show_docstring:
            print(color_scheme, spacing, filename, spacing)
            print('"{}"'.format(display_info))
            print()
        elif args.show_help:
            print(color_scheme, spacing, filename, spacing)
            print(display_info)
            print()
        else:
            print(color_scheme+filename, '"{}"'.format(display_info))

        counter += 1

    if not matches:
        print('No results found!')


def _test_matching_keywords(keywords, texts):
    """Test multiple texts for matching keywords. Returns True only if all
    keywords are present in any of the texts.

    Parameters
    ----------
    keywords : Iterable of str
        Keywords to test for.
    texts : Iterable of str
        Strings that should contain the keywords.

    Returns
    -------
    True if all keywords were found in at least one of the texts.

    """
    matches = []
    for key in keywords:
        key_match = False
        for text in texts:
            if key.lower() in text.lower():
                key_match = True
                break
        matches.append(key_match)

    return np.all(matches)


def _extract_first_sentence(text):
    """Extract the first sentence of a string by finding the first dot. If
    there is no dot, return the full string.

    Parameters
    ----------
    text : str
        Text to parse.

    Returns
    -------
    first_sentence : str
        The first sentence, or the full text if no dot was found.

    """
    first_dot_idx = text.find('.') + 1 or None
    sentence = text[:first_dot_idx]
    return sentence


def _get_docstring(script):
    """Extract a python file's docstring from a filepath.

    Parameters
    ----------
    script : str
        Path to python file

    Returns
    -------
    docstring : str
        The file docstring, or an empty string if there was no docstring.
    """
    with open(script, 'r') as reader:
        file_contents = reader.read()
    module = ast.parse(file_contents)
    docstring = ast.get_docstring(module) or ''
    return docstring


if __name__ == '__main__':
    main()
