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

Verbosity Options:
- If the `-v` option is provided, the script will display the first sentence
  of the docstring for each matching script.
- If the `-v DEBUG` option is provided, the script will display the full
  docstring for each matching script.

Keywords Highlighting:
- When displaying the docstrings, the script highlights the found keywords in
red.

Examples:
- scil_search_keywords.py tractogram filtering
- scil_search_keywords.py "Spherical Harmonics"
- scil_search_keywords.py --no_synonyms "Spherical Harmonics"
- scil_search_keywords.py --search_category tractogram
- scil_search_keywords.py -v sh
- scil_search_keywords.py -v DEBUG sh
"""

import argparse
import logging
import pathlib
import shutil

try:
    import nltk
    nltk.download('punkt', quiet=True)
except ImportError:
    raise ImportError("You must install the 'nltk' package to use this script."
                      "Please run 'pip install nltk'.")

from colorama import Fore, Style
import json

from scilpy.utils.scilpy_bot import (
    _get_docstring_from_script_path, _stem_keywords,
    _stem_phrase, _generate_help_files,
    _get_synonyms, _extract_keywords_and_phrases,
    _calculate_score, _make_title, prompt_user_for_object,
    _split_first_sentence, _highlight_keywords
)
from scilpy.utils.scilpy_bot import SPACING_LEN, VOCAB_FILE_PATH
from scilpy.io.utils import add_verbose_arg

nltk.download('punkt', quiet=True)


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('keywords', nargs='+',
                   help='Search the provided list of keywords.')

    p.add_argument('--search_category', action='store_true',
                   help='Search within a specific category of scripts.')
    p.add_argument('--no_synonyms', action='store_true',
                   help='Search without using synonyms.')

    p.add_argument('--regenerate_help_files', action='store_true',
                   help='Regenerate help files for all scripts.')

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
    else:
        selected_object = ''

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

    if args.regenerate_help_files:
        shutil.rmtree(hidden_dir)

    if not hidden_dir.exists():
        hidden_dir.mkdir()
        logging.info('This is your first time running this script. '
                     'Generating help files may take a few minutes\n '
                     'Please be patient, subsequent searches will be faster.')

    _generate_help_files()

    scores = {}

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
        Returns
        -------
        None
            Just updates the global `matches` and `scores` lists/dictionaries.
        """
        for key, value in score_details.items():
            if value == 0:
                continue

            if filename not in scores:
                scores[filename] = {key: value}
            elif key not in scores[filename]:
                scores[filename].update({key: value})
            else:
                scores[filename][key] += value

        return

    for script in sorted(hidden_dir.glob(f'scil_{selected_object}*.help')):
        filename = script.stem

        # Search through the docstring
        with open(script, 'r') as f:
            search_text = f.read()

        score_details = _calculate_score(
            stemmed_keywords, stemmed_phrases, search_text, filename=filename)
        update_matches_and_scores(filename, score_details)

    # Search in keywords file
    with open(VOCAB_FILE_PATH, 'r') as f:
        vocab_data = json.load(f)

    for script in vocab_data['scripts']:
        script_name = script['name']
        script_keywords = script['keywords']
        search_text = ' '.join(script_keywords)
        score_details = _calculate_score(
            stemmed_keywords, stemmed_phrases, search_text, script_name)
        update_matches_and_scores(script_name, score_details)

    # Search in synonyms file if not args.no_synonyms is not specified
    if not args.no_synonyms:
        full_list_to_verify = set(stemmed_keywords + stemmed_phrases)
        for keyword in full_list_to_verify:
            synonyms = _get_synonyms(keyword, vocab_data['synonyms'])

            for script in sorted(hidden_dir.glob(f'scil_{selected_object}*.help')):
                score_details = {}
                filename = script.stem

                with open(script, 'r') as f:
                    search_text = f.read()

                for synonym in synonyms:
                    if filename in scores and synonym in scores[filename]:
                        continue

                    if ' ' in synonym:
                        stemmed_phrases = [synonym]
                        stemmed_keywords = []
                    else:
                        stemmed_keywords = [synonym]
                        stemmed_phrases = []

                    score_details = _calculate_score(stemmed_keywords,
                                                     stemmed_phrases,
                                                     search_text,
                                                     script_name,
                                                     suffix=' (synonyms)')

                    # Directly update scores dictionary
                    if filename in scores:
                        scores[filename].update(score_details)
                    else:
                        scores[filename] = score_details

    matches = list(scores.keys())

    if not matches:
        logging.info(_make_title(' No results found! '))
        return

    total_scores = {match: sum(scores[match].values()) for match in matches}
    sorted_matches = sorted(total_scores, key=total_scores.get)

    # Sort matches by score and display them
    logging.info(_make_title(' Results Ordered by Score '))
    for match in sorted_matches:
        if total_scores[match] == 0:
            continue
        logging.info(f"{Fore.BLUE}{Style.BRIGHT}{match}{Style.RESET_ALL}")

        for word, score in scores[match].items():
            original_word = keyword_mapping.get(
                word, phrase_mapping.get(word, word))
            logging.info(
                f"{Fore.GREEN}Occurrence of '{original_word}': {score}{Style.RESET_ALL}")

        # Highlight keywords based on verbosity level
        with open(hidden_dir / f'{match}.help', 'r') as f:
            docstrings = f.read()
        highlighted_docstring = _highlight_keywords(docstrings,
                                                    stemmed_keywords)
        if args.verbose == 'INFO':
            first_sentence = _split_first_sentence(
                highlighted_docstring)[0]
            logging.info(f"{first_sentence.strip()}")
        elif args.verbose == 'DEBUG':
            logging.debug(f"{highlighted_docstring.strip()}")
        logging.info(
            f"{Fore.RED}Total Score: {total_scores[match]}{Style.RESET_ALL}")
        logging.info(f"{Fore.BLUE}{'=' * SPACING_LEN}")
        logging.info("\n")

    logging.info(_make_title(
        ' Results Ordered by Score (Best results at the bottom) '))


if __name__ == '__main__':
    main()
