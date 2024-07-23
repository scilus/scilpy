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
import logging
import pathlib
import nltk
from colorama import init, Fore, Style
import json

from scilpy.utils.scilpy_bot import (
    _get_docstring_from_script_path, _split_first_sentence, _stem_keywords,
    _stem_text, _stem_phrase, _generate_help_files, _highlight_keywords,
    _get_synonyms, _extract_keywords_and_phrases, _calculate_score, _make_title
)

from scilpy.utils.scilpy_bot import SPACING_LEN, KEYWORDS_FILE_PATH, SYNONYMS_FILE_PATH
from scilpy.io.utils import add_verbose_arg

nltk.download('punkt', quiet=True)

OBJECTS = [
    'aodf', 'bids', 'bingham', 'btensor', 'bundle', 'connectivity', 'denoising',
    'dki', 'dti','dwi', 'fodf', 'freewater', 'frf', 'gradients', 'header', 'json',
    'labels', 'lesions', 'mti', 'NODDI', 'sh', 'surface', 'tracking',
    'tractogram', 'viz', 'volume', 'qball', 'rgb', 'lesions'
]

def prompt_user_for_object():
    print("Available objects:")
    for idx, obj in enumerate(OBJECTS):
        print(f"{idx + 1}. {obj}")
    while True:
        try:
            choice = int(input("Choose the object you want to work on (enter the number): "))
            if 1 <= choice <= len(OBJECTS):
                return OBJECTS[choice - 1]
            else:
                print(f"Please enter a number between 1 and {len(OBJECTS)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    #p.add_argument('--object', choices=OBJECTS, required=True,
    #              help='Choose the object you want to work on.' )
    p.add_argument('keywords', nargs='+',
                   help='Search the provided list of keywords.')

    p.add_argument('--full_parser', action='store_true',
                   help='Display the full script argparser help.')
    
    p.add_argument('--search_category', action='store_true',
                   help='Search within a specific category of scripts.')

    add_verbose_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    if args.verbose == "WARNING":
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    selected_object = None
    if args.search_category:
        selected_object = prompt_user_for_object()

    keywords, phrases = _extract_keywords_and_phrases(args.keywords)
    stemmed_keywords = _stem_keywords(args.keywords)
    stemmed_phrases = [_stem_phrase(phrase) for phrase in phrases]

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
    scores = {}

    # Determine the pattern to search for
    search_pattern = f'scil_{"{}_" if selected_object else ""}*.py'

    # Search through the docstring
    logging.info(f"Searching through docstrings for '{selected_object}' scripts...")
    for script in sorted(script_dir.glob(search_pattern.format(selected_object))):
        #Remove the .py extension
        filename = script.stem
        if filename == '__init__' or filename =='scil_search_keywords':
            continue

        search_text = _get_docstring_from_script_path(str(script))
        score_details = _calculate_score(stemmed_keywords, stemmed_phrases, search_text, filename=filename)

        if score_details['total_score']  > 0:        
            matches.append(filename)
            scores[filename] = score_details

            search_text = search_text or 'No docstring available!'

            display_filename = filename + '.py'
            display_short_info, display_long_info = _split_first_sentence(
                search_text)

            # Highlight found keywords using colorama
            display_short_info = _highlight_keywords(display_short_info, stemmed_keywords)
            display_long_info = _highlight_keywords(display_long_info, stemmed_keywords)

            # Print everything
            logging.info(f"{Fore.BLUE}{Style.BRIGHT}{display_filename}{Style.RESET_ALL}")
            logging.info(display_short_info)
            logging.debug(display_long_info)
            for word, score in score_details.items():
                if word != 'total_score':
                    logging.info(f"{Fore.GREEN}Occurence of '{word}': {score}{Style.RESET_ALL}")
            logging.info(f"Total Score: {score_details['total_score']}")
            logging.info(f"{Fore.BLUE}{'=' * SPACING_LEN}")
            logging.info("\n")

    # If no matches found in docstrings, check in the help files 

    if not matches: 
        logging.info(f"No matches found in docstrings. Searching through help files for '{selected_object}' scripts...") 
        for help_file in sorted(hidden_dir.glob(search_pattern.format(selected_object))): #Use precomputed help files
            script_name = pathlib.Path(help_file.stem).stem
            with open(help_file, 'r') as f:
                search_text = f.read()

            score_details = _calculate_score(stemmed_keywords, stemmed_phrases, search_text, script_name)

            if score_details['total_score'] > 0:
                matches.append(script_name)
                scores[script_name] = score_details

                search_text = search_text or 'No docstring available!'

                display_filename = script_name + '.py'
                display_short_info, display_long_info = _split_first_sentence(
                    search_text)

                # Print everything
                logging.info(f"{Fore.BLUE}{Style.BRIGHT}{display_filename}{Style.RESET_ALL}")
                logging.info(display_short_info)
                logging.debug(display_long_info)
                for word, score in score_details.items():
                    if word != 'total_score':
                        logging.info(f"{Fore.GREEN}Occurence of '{word}': {score}{Style.RESET_ALL}")
                logging.info(f"Total Score: {score_details['total_score']}")
                logging.info(f"{Fore.BLUE}{'=' * SPACING_LEN}")
                logging.info("\n")

    # If no matches found, check in the keywords file
    with open(KEYWORDS_FILE_PATH, 'r') as f:
        keywords_data = json.load(f)

    if not matches:
        logging.info("No matches found in help files. Searching by script keywords...")
        for script in keywords_data['scripts']:
            script_name = script['name']
            if selected_object and not script_name.startswith(f'scil_{selected_object}_'):
                continue
            script_keywords = script['keywords']
            score_details  = _calculate_score(stemmed_keywords, stemmed_phrases,' '.join(script_keywords), script_name)

            if score_details['total_score'] > 0:
                matches.append(script_name)
                scores[script_name] = score_details

                display_filename = script_name + '.py'
                first_sentence, _ = _split_first_sentence(search_text)
                logging.info(f"{Fore.BLUE}{'=' * SPACING_LEN}")
                logging.info(f"{Fore.BLUE}{Style.BRIGHT}{display_filename}{Style.RESET_ALL}: {first_sentence}")
                logging.info("\n")
 

    # If still no matches found, check for synonyms in the synonyms file
    with open(SYNONYMS_FILE_PATH, 'r') as f:
        synonyms_data = json.load(f)
        
    if not matches:
        logging.info("No matches found by script keywords. Searching by synonyms...")
        for keyword in args.keywords:
            synonyms = _get_synonyms(keyword, synonyms_data)
            for script in sorted(script_dir.glob(search_pattern.format(selected_object))):
                filename = script.stem
                if filename == '__init__' or filename == 'scil_search_keywords':
                    continue
                search_text = _get_docstring_from_script_path(str(script))
                if any(synonym in search_text for synonym in synonyms):
                    matches.append(filename)
                    scores[filename] = _calculate_score(synonyms,[], search_text, filename)
                    first_sentence, _ = _split_first_sentence(search_text)
                    display_filename = filename + '.py'
                    logging.info(f"{Fore.BLUE}{'=' * SPACING_LEN}")
                    logging.info(f"{Fore.BLUE}{Style.BRIGHT}{filename}{Style.RESET_ALL}: {first_sentence}")
                    logging.info("\n")

    if not matches:
        logging.info(_make_title(' No results found! '))

    """# Sort matches by score and print them
    else:
        sorted_matches = sorted(matches, key=lambda x: scores[x]['total_score'], reverse=True)
        logging.info(_make_title(' Results Ordered by Score '))
        for match in sorted_matches:
            display_filename = match + '.py'
            logging.info(f"{Fore.BLUE}{Style.BRIGHT}{display_filename}{Style.RESET_ALL}: Score = {scores[match]['total_score']}")"""

    # Display full argparser if --full_parser is used
    if args.full_parser:
        for script in sorted(script_dir.glob('scil_{}_*.py'.format(selected_object))):
            filename = script.stem
            if filename == '__init__' or filename == 'scil_search_keywords':
                continue
            help_file = hidden_dir / f"{filename}.py.help"
            if help_file.exists():
                with open(help_file, 'r') as f:
                    display_filename = filename + '.py'
                    logging.info(f"{Fore.BLUE}{Style.BRIGHT}{display_filename}{Style.RESET_ALL}")
                    logging.info(f.read())
                    logging.info(f"{Fore.BLUE}{'=' * SPACING_LEN}")
                    logging.info("\n")



if __name__ == '__main__':
    main()