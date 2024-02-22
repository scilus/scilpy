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

import numpy as np

from scilpy.io.utils import add_verbose_arg

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

    add_verbose_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    if args.verbose == "WARNING":
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    # Use directory of this script, should work with most installation setups
    script_dir = pathlib.Path(__file__).parent
    matches = []

    keywords_regexes = [re.compile('(' + re.escape(kw) + ')', re.IGNORECASE)
                        for kw in args.keywords]

    for script in sorted(script_dir.glob('*.py')):
        filename = script.name
        if filename == '__init__.py':
            continue

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
                search_text = _get_docstring_from_script_path(str(script))
        else:
            # Fetch the docstring
            search_text = _get_docstring_from_script_path(str(script))

        # Test intersection of all keywords, either in filename or docstring
        if not _test_matching_keywords(args.keywords, [filename, search_text]):
            continue

        matches.append(filename)
        search_text = search_text or 'No docstring available!'

        display_filename = filename
        display_short_info, display_long_info = _split_first_sentence(
            search_text)

        # NOTE: It is important to do the formatting before adding color style,
        # because python does not ignore ANSI color codes, and will count them
        # as characters!

        # Center text, add spacing and make BOLD
        header = _make_title(" {} ".format(display_filename))
        footer = _make_title(" End of {} ".format(display_filename))

        # Highlight found keywords using ANSI color codes
        colored_keyword = '{}\\1{}'.format(RED + BOLD, END_COLOR)
        for regex in keywords_regexes:
            header = regex.sub(colored_keyword, header)
            footer = regex.sub(colored_keyword, footer)
            display_short_info = regex.sub(colored_keyword, display_short_info)
            display_long_info = regex.sub(colored_keyword, display_long_info)

        # Restore BOLD in header/footer after matching keywords, and make sure
        # to add a END_COLOR at the end.
        header = header.replace(END_COLOR, END_COLOR + BOLD) + END_COLOR
        footer = footer.replace(END_COLOR, END_COLOR + BOLD) + END_COLOR

        # Print everything
        logging.info(header)
        if error_msg:
            logging.info(RED + BOLD + error_msg + END_COLOR)
        logging.info(display_short_info)
        logging.debug(display_long_info)
        logging.info(footer)
        logging.info("\n")

    if not matches:
        logging.info(_make_title(' No results found! '))


def _make_title(text):
    return BOLD + text.center(SPACING_LEN, SPACING_CHAR) + END_COLOR


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


def _get_docstring_from_script_path(script):
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


def _split_first_sentence(text):
    """Split the first sentence from the rest of a string by finding the first
    dot or newline. If there is no dot or newline, return the full string as
    the first sentence, and None as the remaining text.

    Parameters
    ----------
    text : str
        Text to parse.

    Returns
    -------
    first_sentence : str
        The first sentence, or the full text if no dot or newline was found.
    remaining : str
        Everything after the first sentence.

    """
    candidates = ['. ', '.\n']
    sentence_idx = -1
    for candidate in candidates:
        idx = text.find(candidate)
        if idx != -1 and idx < sentence_idx or sentence_idx == -1:
            sentence_idx = idx

    split_idx = (sentence_idx + 1) or None
    sentence = text[:split_idx]
    remaining = text[split_idx:] if split_idx else ""
    return sentence, remaining


if __name__ == '__main__':
    main()
