#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Search through all SCILPY scripts and their docstrings to find matches for the provided keywords.
The search will be performed across script names, docstrings, help files, keywords, and optionally synonyms.
The output will list the matching filenames along with the occurrences of each keyword, and their total score.

- By default, the search includes synonyms for the keywords.
- Use --no_synonyms to exclude synonyms from the search.
- Use --search_category to limit the search to a specific category of scripts.
- Use --verbose to display the full docstring.
- Words enclosed in quotes will be searched as phrases, ensuring the words appear next to each other in the text.


Examples:
    scil_search_keywords.py tractogram filtering
    scil_search_keywords.py --search_parser tractogram filtering -v
    scil_search_keywords.py "Spherical Harmonics" convert
    scil_search_keywords.py --no_synonyms tractogram filtering
    scil_search_keywords.py --search_category --verbose tractogram filtering
"""

import argparse
import logging
import pathlib
import nltk
from colorama import init, Fore, Style
import json

from scilpy.utils.scilpy_bot import (
    _get_docstring_from_script_path, _split_first_sentence, _stem_keywords, _stem_phrase, _generate_help_files,
    _get_synonyms, _extract_keywords_and_phrases, _calculate_score, _make_title, prompt_user_for_object
)

from scilpy.utils.scilpy_bot import SPACING_LEN, KEYWORDS_FILE_PATH, SYNONYMS_FILE_PATH
from scilpy.io.utils import add_verbose_arg

nltk.download('punkt', quiet=True)


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    #p.add_argument('--object', choices=OBJECTS, required=True,
    #              help='Choose the object you want to work on.' )
    p.add_argument('keywords', nargs='+',
                   help='Search the provided list of keywords.')
    
    p.add_argument('--search_category', action='store_true',
                   help='Search within a specific category of scripts.')
    
    p.add_argument('--no_synonyms', action='store_true',
               help='Search without using synonyms.')


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
    stemmed_keywords = _stem_keywords(keywords)
    stemmed_phrases = [_stem_phrase(phrase) for phrase in phrases]

    # Create a mapping of stemmed to original keywords(will be needed to display the occurence of the keywords)
    keyword_mapping = {stem: orig for orig, stem in zip(keywords, stemmed_keywords)}
    phrase_mapping = {stem: orig for orig, stem in zip(phrases, stemmed_phrases)}

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

    # pattern to search for
    search_pattern = f'scil_{"{}_" if selected_object else ""}*.py'

    def update_matches_and_scores(filename, score_details):
        if score_details['total_score'] > 0:
            if filename not in matches:
                matches.append(filename)
                scores[filename] = score_details
            else:
                for key, value in score_details.items():
                    if key != 'total_score':
                        scores[filename][key] = scores[filename].get(key, 0) + value
                scores[filename]['total_score'] += score_details['total_score']

    for script in sorted(script_dir.glob(search_pattern.format(selected_object))):
        filename = script.stem
        if filename == '__init__' or filename =='scil_search_keywords':
            continue
        
        # Search through the docstring
        search_text = _get_docstring_from_script_path(str(script))
        score_details = _calculate_score(stemmed_keywords, stemmed_phrases, search_text, filename=filename)
        update_matches_and_scores(filename, score_details)
        

        # Search in help files
        help_file = hidden_dir / f"{filename}.py.help"
        if help_file.exists():
            with open(help_file, 'r') as f:
                search_text = f.read()
            score_details = _calculate_score(stemmed_keywords, stemmed_phrases, search_text, filename=filename)
            update_matches_and_scores(filename, score_details)

    # Search in keywords file
    with open(KEYWORDS_FILE_PATH, 'r') as f:
        keywords_data = json.load(f)

    for script in keywords_data['scripts']:
        script_name = script['name']
        if selected_object and not script_name.startswith(f'scil_{selected_object}_'):
            continue
        script_keywords = script['keywords']
        search_text = ' '.join(script_keywords)
        score_details = _calculate_score(stemmed_keywords, stemmed_phrases, search_text, script_name)
        update_matches_and_scores(script_name, score_details)


    # Search in synonyms file if not args.no_synonyms is not specified
    if not args.no_synonyms:
        with open(SYNONYMS_FILE_PATH, 'r') as f:
            synonyms_data = json.load(f)

        # Create a mapping of synonyms to their original keywords
        synonym_to_keyword = {}   
        for keyword in args.keywords:
            synonyms = _get_synonyms(keyword, synonyms_data)
            for synonym in synonyms:
                synonym_to_keyword[synonym] = keyword
            
            for script in sorted(script_dir.glob(search_pattern.format(selected_object))):
                filename = script.stem
                if filename == '__init__' or filename == 'scil_search_keywords':
                    continue
                search_text = _get_docstring_from_script_path(str(script))
                synonym_score = 0
                for synonym in synonyms:
                    if synonym in search_text:
                        synonym_score += search_text.count(synonym)
                if synonym_score > 0:
                    if filename not in scores:
                        scores[filename] = {'total_score': 0}
                        matches.append(filename) 
                    scores[filename][keyword] = scores[filename].get(keyword, 0) + synonym_score
                    scores[filename]['total_score'] += synonym_score
    
    if not matches:
        logging.info(_make_title(' No results found! '))

    # Sort matches by score and print them
    else:
        sorted_matches = sorted(matches, key=lambda x: scores[x]['total_score'], reverse=True)

        logging.info(_make_title(' Results Ordered by Score '))
        for match in sorted_matches:
            #display_filename = match + '.py'
            logging.info(f"{Fore.BLUE}{Style.BRIGHT}{match}{Style.RESET_ALL}")
            for word, score in scores[match].items():
                if word != 'total_score':
                    original_word = keyword_mapping.get(word, phrase_mapping.get(word, word))
                    logging.info(f"{Fore.GREEN}Occurrence of '{original_word}': {score}{Style.RESET_ALL}")
            logging.info(f"Total Score: {scores[match]['total_score']}")
            logging.info(f"{Fore.BLUE}{'=' * SPACING_LEN}")
            logging.info("\n")



if __name__ == '__main__':
    main()