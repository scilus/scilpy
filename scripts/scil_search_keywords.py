#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Search through all of SCILPY scripts and their docstrings. The output of the
search will be the intersection of all provided keywords, found either in the
script name or in its docstring.

Examples:
    scil_search_keywords.py tractogram filtering
    scil_search_keywords.py --search_parser tractogram filtering
"""

import argparse
import ast
import logging
import pathlib
import re
import subprocess

import numpy as np

RED = '\033[31m'
BOLD = '\033[1m'
END_COLOR = '\033[0m'
SPACING_CHAR = '='
SPACING_LEN = 80


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('keywords', nargs='+',
                   help='Search the provided list of keywords.')

    p.add_argument('--search_parser', action='store_true',
                   help='Search through and display the full script argparser '
                        'instead of looking only at the docstring. (warning: '
                        'much slower).')
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Use directory of this script, should work with most installation setups
    script_dir = pathlib.Path(__file__).parent
    matches = []

    kw_subs = [re.compile('(' + re.escape(kw) + ')', re.IGNORECASE)
               for kw in args.keywords]

    for script in sorted(script_dir.glob('*.py')):
        filename = script.name

        # Skip this script
        if filename == pathlib.Path(__file__).name:
            continue

        error_msg = ""
        if args.search_parser:
            # Run the script's argparser, and catch the output in case there
            # is an error, such as ModuleNotFoundException.
            sub = subprocess.run(['{}'.format(script.absolute()), '--help'],
                                 capture_output=True)
            search_text = sub.stdout.decode("utf-8")
            if sub.stderr:
                # Fall back on the docstring in case of error
                error_msg = "There was an error executing script parser, " \
                            "searching through docstring instead...\n\n"
                search_text = _get_docstring(str(script))
        else:
            # Fetch the docstring
            search_text = _get_docstring(str(script))

        # Test intersection of all keywords, either in filename or docstring
        if not _test_matching_keywords(args.keywords, [filename, search_text]):
            continue

        matches.append(filename)
        search_text = search_text or 'No docstring available!'

        new_key = '{}\\1{}'.format(RED + BOLD, END_COLOR)

        display_text = search_text
        for regex in kw_subs:
            # Highlight found keywords
            filename = regex.sub(new_key, filename)
            display_text = regex.sub(new_key, display_text)

        # Keep title in BOLD after matching keyword
        filename = filename.replace(END_COLOR, END_COLOR + BOLD)

        logging.info(_colorize(" {} ".format(filename), BOLD)
                     .center(SPACING_LEN, SPACING_CHAR))
        if error_msg:
            logging.info(RED + BOLD + error_msg + END_COLOR)
        logging.info(display_text)
        logging.info(_colorize(" End of {} ".format(filename), BOLD)
                     .center(SPACING_LEN, SPACING_CHAR))
        logging.info("\n")

    if not matches:
        logging.info('No results found!')


def _colorize(text, color):
    return color + text + END_COLOR


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
