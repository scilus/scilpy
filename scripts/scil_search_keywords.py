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
import nltk
from nltk.stem import PorterStemmer
from colorama import init, Fore, Style
import json
from pathlib import Path
from scilpy.io.utils import add_verbose_arg

nltk.download('punkt', quiet=True)

init(autoreset=True)

RED = '\033[31m'
BOLD = '\033[1m'
END_COLOR = '\033[0m'
SPACING_CHAR = '='
SPACING_LEN = 80

BASE_DIR =  Path(__file__).parent.parent
JSON_FILE_PATH = BASE_DIR / 'scilpy-bot-scripts'/'json_files'/'knowledge_base.json'

stemmer = PorterStemmer()

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

    stemmed_keywords = _stem_keywords(args.keywords)

    # Use directory of this script, should work with most installation setups
    script_dir = pathlib.Path(__file__).parent
    hidden_dir = script_dir / '.hidden'

    if not hidden_dir.exists():
        hidden_dir.mkdir()
        logging.info('This is your first time running this script.\n'
                     'Generating help files may take a few minutes, please be patient.\n'
                     'Subsequent searches will be much faster.\n'
                     'Generating help files....')
        _generate_help_files()


    matches = []

    keywords_regexes = [re.compile('(' + re.escape(kw) + ')', re.IGNORECASE)
                        for kw in args.keywords]

    # Search through the argparser instead of the docstring        
    if args.search_parser: 
        #Use precomputed help files
        for help_file in sorted(hidden_dir.glob('*.help')):
            script_name = pathlib.Path(help_file.stem).stem
            with open(help_file, 'r') as f:
                search_text = f.read()

            
        # Test intersection of all keywords, either in filename or docstring
            if not _contains_stemmed_keywords(stemmed_keywords, search_text, script_name):
                continue

            matches.append(script_name)
            search_text = search_text or 'No docstring available!'

            display_filename = script_name
            display_short_info, display_long_info = _split_first_sentence(
                search_text)

            # Highlight found keywords using colorama
            for keyword in args.keywords:
                display_short_info = display_short_info.replace(keyword, f'{Fore.RED}{Style.BRIGHT}{keyword}{Style.RESET_ALL}')
                display_long_info = display_long_info.replace(keyword, f'{Fore.RED}{Style.BRIGHT}{keyword}{Style.RESET_ALL}')

            # Print everything
            logging.info(f"{Fore.BLUE}{Style.BRIGHT}{display_filename}{Style.RESET_ALL}")
            logging.info(display_short_info)
            logging.debug(display_long_info)
            logging.info(f"{Fore.BLUE}{'=' * SPACING_LEN}")
            logging.info("\n")
    
    # Search through the docstring instead of the argparser
    else:
        for script in sorted(script_dir.glob('*.py')):
            #Remove the .py extension
            filename = script.stem
            if filename == '__init__' or filename =='scil_search_keywords':
                continue
            
            search_text = _get_docstring_from_script_path(str(script))

            # Test intersection of all keywords, either in filename or docstring
            if not _contains_stemmed_keywords(stemmed_keywords, search_text, filename):
                continue

            matches.append(filename)
            search_text = search_text or 'No docstring available!'

            display_filename = filename
            display_short_info, display_long_info = _split_first_sentence(
                search_text)

            # Highlight found keywords using colorama
            display_short_info = _highlight_keywords(display_short_info, stemmed_keywords)
            display_long_info = _highlight_keywords(display_long_info, stemmed_keywords)

            # Print everything
            logging.info(f"{Fore.BLUE}{Style.BRIGHT}{display_filename}{Style.RESET_ALL}")
            logging.info(display_short_info)
            logging.debug(display_long_info)
            logging.info(f"{Fore.BLUE}{'=' * SPACING_LEN}")
            logging.info("\n")

            
    if not matches:
        logging.info(_make_title(' No such keyword found! Let\'s look for synonyms... '))
        scripts = load_json(JSON_FILE_PATH)
        # Search for synonyms if no matches found
        matches = search_keywords_in_synonyms(scripts, args.keywords)
        if matches:
            logging.info(f"Found {len(matches)} scripts with matching synonyms. Displaying first 5:")
            for match in matches[:5]:
                logging.info(f"{Fore.BLUE}{Style.BRIGHT}{match['name']}{Style.RESET_ALL}")
                display_short_info, _ = _split_first_sentence(match.get('docstring', 'No docstring available!'))
                display_short_info = _highlight_keywords(display_short_info, stemmed_keywords)
                logging.info(display_short_info)
        else:
            logging.info(_make_title(' No results found in synonyms either! '))


def _make_title(text):
    return f'{Fore.BLUE}{Style.BRIGHT}{text.center(SPACING_LEN, SPACING_CHAR)}{Style.RESET_ALL}'


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

def _stem_keywords(keywords):
    return [stemmer.stem(keyword) for keyword in keywords]

def _stem_text(text):
    words = nltk.word_tokenize(text)
    return ' '.join([stemmer.stem(word) for word in words])

def _contains_stemmed_keywords(stemmed_keywords,text, filename):
    stemmed_text = _stem_text(text)
    stemmed_filename = _stem_text(filename)
    return all([stem in stemmed_text or stem in stemmed_filename for stem in stemmed_keywords])

def _generate_help_files():
    """Call the external script generate_help_files to generate help files
    """
    script_path = pathlib.Path(__file__).parent.parent / 'scilpy-bot-scripts'/'generate_help_files.py'
    #calling the extrernal script generate_help_files
    subprocess.run(['python', script_path], check=True)
    
def _highlight_keywords(text, stemmed_keywords):
    """Highlight the stemmed keywords in the given text using colorama."""
    words = text.split()
    highlighted_text = []
    for word in words:
        stemmed_word = stemmer.stem(word)
        if stemmed_word in stemmed_keywords:
            highlighted_text.append(f'{Fore.RED}{Style.BRIGHT}{word}{Style.RESET_ALL}')
        else:
            highlighted_text.append(word)
    return ' '.join(highlighted_text)

def load_json(json_filepath):
    with open(json_filepath, 'r', encoding='utf-8') as file:
        return json.load(file)

def search_keywords_in_synonyms(scripts, keywords):
    matches = []
    for script in scripts['scripts']:
        for synonym_list in script.get('synonyms', []):
            if any(keyword in synonym_list for keyword in keywords):
                matches.append(script)
                break
    return matches

if __name__ == '__main__':
    main()
