#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Search through all SCILPY scripts and their docstrings to find matches for the
provided keywords.
The search will be performed across script names, docstrings, help files,
keywords, and optionally synonyms.
The output will list the matching filenames along with the occurrences of each
keyword, and their total score.

- By default, the search includes synonyms for the keywords.
- Use --no_synonyms to exclude synonyms from the search.
- Use --search_category to limit the search to a specific category of scripts.
- Words enclosed in quotes will be searched as phrases, ensuring the words
appear next to each other in the text.


Examples:
- scil_search_keywords.py tractogram filtering
- scil_search_keywords.py "Spherical Harmonics" convert
- scil_search_keywords.py --no_synonyms tractogram filtering
- scil_search_keywords.py --search_category tractogram filtering
"""

import argparse
import logging
import pathlib

try:
    import nltk
    nltk.download('punkt', quiet=True)
except ImportError:
    print("You must install the 'nltk' package to use this script."
          "Please run 'pip install nltk'.")
    exit(1)

from colorama import Fore, Style
import json

from scilpy.utils.scilpy_bot import (
    _get_docstring_from_script_path, _stem_keywords,
    _stem_phrase, _generate_help_files,
    _get_synonyms, _extract_keywords_and_phrases,
    _calculate_score, _make_title, prompt_user_for_object
)
from scilpy.utils.scilpy_bot import SPACING_LEN, VOCAB_FILE_PATH
from scilpy.io.utils import add_verbose_arg

nltk.download('punkt', quiet=True)


def _initialize_logging(verbosity):
    logging.basicConfig(level=logging.WARNING)
    if verbosity == 'INFO':
        logging.getLogger().setLevel(logging.INFO)
    elif verbosity == 'DEBUG':
        logging.getLogger().setLevel(logging.DEBUG)


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

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

    # keywords are single words. Phrases are composed keywords
    keywords, phrases = _extract_keywords_and_phrases(args.keywords)
    stemmed_keywords = _stem_keywords(keywords)
    stemmed_phrases = [_stem_phrase(phrase) for phrase in phrases]

    # Create a mapping of stemmed to original keywords
    # This will be needed to display the occurence of the keywords
    keyword_mapping = {stem: orig for orig,
                       stem in zip(keywords, stemmed_keywords)}
    phrase_mapping = {stem: orig for orig,
                      stem in zip(phrases, stemmed_phrases)}

    script_dir = pathlib.Path(__file__).parent
    hidden_dir = script_dir / '.hidden'

    if not hidden_dir.exists():
        hidden_dir.mkdir()
        logging.info('This is your first time running this script.\n'
                     'Generating help files may take a few minutes,'
                     'please be patient.\n'
                     'Subsequent searches will be much faster.')
        _generate_help_files()

    matches = []
    scores = {}

    # pattern to search for
    search_pattern = f'scil_{"{}_" if selected_object else ""}*.py'

    def update_matches_and_scores(filename, score_details):
        """
        Update the matches and scores for the given filename based
        on the score details.

        Parameters
        ----------
        filename : str
            The name of the script file being analyzed.
        score_details : dict
            A dictionary containing the scores for the keywords
                and phrases found in the script.
            This dictionary should have a 'total_score' key
                indicating the cumulative score.

        Returns
        -------
        None
            Just updates the global `matches` and `scores` lists/dictionaries.
        """
        if score_details['total_score'] > 0:
            if filename not in matches:
                matches.append(filename)
                scores[filename] = score_details
            else:
                for key, value in score_details.items():
                    if key != 'total_score':
                        scores[filename][key] = scores[filename].get(
                            key, 0) + value
                scores[filename]['total_score'] += score_details['total_score']

    for script in sorted(script_dir.glob(search_pattern.format(selected_object))):
        filename = script.stem
        if filename == '__init__' or filename == 'scil_search_keywords':
            continue

        # Search through the docstring
        search_text = _get_docstring_from_script_path(str(script))
        score_details = _calculate_score(
            stemmed_keywords, stemmed_phrases, search_text, filename=filename)
        update_matches_and_scores(filename, score_details)

        # Search in help files
        help_file = hidden_dir / f"{filename}.py.help"
        if help_file.exists():
            with open(help_file, 'r') as f:
                search_text = f.read()
            score_details = _calculate_score(
                stemmed_keywords, stemmed_phrases,
                search_text, filename=filename)
            update_matches_and_scores(filename, score_details)

    # Search in keywords file
    with open(VOCAB_FILE_PATH, 'r') as f:
        vocab_data = json.load(f)

    for script in vocab_data['scripts']:
        script_name = script['name']
        if selected_object and not script_name.startswith(f'scil_{selected_object}_'):
            continue
        script_keywords = script['keywords']
        search_text = ' '.join(script_keywords)
        score_details = _calculate_score(
            stemmed_keywords, stemmed_phrases, search_text, script_name)
        update_matches_and_scores(script_name, score_details)

    # Search in synonyms file if not args.no_synonyms is not specified
    if not args.no_synonyms:
        for keyword in keywords + phrases:
            synonyms = _get_synonyms(keyword, vocab_data['synonyms'])
            for script in sorted(script_dir.glob(search_pattern.format(selected_object))):
                filename = script.stem
                if filename == '__init__' or filename == 'scil_search_keywords':
                    continue
                search_text = _get_docstring_from_script_path(str(script))
                # Initialize or get existing score_details for the script
                score_details = scores.get(filename, {'total_score': 0})

                for synonym in synonyms:
                    if synonym in search_text and synonym != keyword:
                        # Update score_details with the count of each synonym found
                        score_details[keyword + ' synonyms'] = score_details.get(
                            keyword + ' synonyms', 0) + search_text.count(synonym)
                        score_details['total_score'] += search_text.count(
                            synonym)

                # Directly update scores dictionary
                scores[filename] = score_details

    if not matches:
        logging.info(_make_title(' No results found! '))

    # Sort matches by score and display them
    else:
        sorted_matches = sorted(
            matches, key=lambda x: scores[x]['total_score'], reverse=False)

        logging.info(_make_title(' Results Ordered by Score '))
        for match in sorted_matches:
            if scores[match]['total_score'] > 0:
                logging.info(f"{Fore.BLUE}{Style.BRIGHT}{match}{Style.RESET_ALL}")

            for word, score in scores[match].items():
                if word != 'total_score':
                    if word.endswith(' synonyms'):
                        logging.info(
                            f"{Fore.GREEN}Occurrence of '{word}': {score}{Style.RESET_ALL}")
                    else:
                        original_word = keyword_mapping.get(
                            word, phrase_mapping.get(word, word))
                        logging.info(
                            f"{Fore.GREEN}Occurrence of '{original_word}': {score}{Style.RESET_ALL}")

            logging.info(f"Total Score: {scores[match]['total_score']}")
            logging.info(f"{Fore.BLUE}{'=' * SPACING_LEN}")
            logging.info("\n")
        logging.info(_make_title(
            ' Results Ordered by Score (Best results at the bottom) '))


if __name__ == '__main__':
    main()
