#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Search through all of SCILPY scripts and their docstrings. The output of the
search will be the intersection of all provided keywords, found either in the
script name or in its docstring.
"""

from argparse import ArgumentParser, RawTextHelpFormatter
import ast
import inspect
import pathlib

import scilpy
from scilpy.io.utils import add_verbose_arg

RED = '\033[31m'
BOLD = '\033[1m'
ENDC = '\033[0m'


def _build_args_parser():
    parser = ArgumentParser(description=__doc__,
                            formatter_class=RawTextHelpFormatter)
    parser.add_argument('keywords', nargs='+',
                        help="Search keywords")
    return parser


def main():
    parser = _build_args_parser()
    add_verbose_arg(parser)
    args = parser.parse_args()

    scilpy_init_file = inspect.getfile(scilpy)  # scilpy/scilpy/__init__.py
    script_dir = pathlib.Path(scilpy_init_file).parent / "../scripts"
    matches = []

    for script in sorted(script_dir.glob('*.py')):
        filename = script.name
        docstring = _get_docstring(str(script))

        if args.verbose:
            # Display full docstring
            display_info = docstring
        else:
            # Extract first sentence by finding the first dot
            display_info = _extract_first_sentence(docstring)

        # Remove newlines
        display_info = display_info.replace("\n", " ")

        # Test intersection of all keywords, either in filename or docstring
        if not _is_matching_keywords(args.keywords, [filename, docstring]):
            continue

        matches.append(filename)

        for key in args.keywords:
            filename = filename.replace(key,
                                        "{}{}{}".format(RED + BOLD, key,
                                                        ENDC))
            display_info = display_info.replace(key,
                                                "{}{}{}".format(RED + BOLD, key,
                                                                ENDC))

        display_info = display_info or "No docstring available!"
        print("===== ", filename, '"{}"'.format(display_info))

    if not matches:
        print("No results found!")


def _is_matching_keywords(keywords, texts):
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
    search_match = True
    for key in keywords:
        for text in texts:
            if not text or key not in text:
                search_match = False
                break

    return search_match


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
    docstring = ast.get_docstring(module) or ""
    return docstring


if __name__ == '__main__':
    main()
