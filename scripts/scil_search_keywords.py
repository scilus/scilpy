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

# TODO harmonize variable names
# TODO add more tests (generic even if code evolves)
# TODO add more comments about the stemming and synonyms
# TODO Order imports and functions alphabetically
# TODO remove useless code if any

import argparse
from colorama import Fore, Style
import json
import logging
import pathlib
import shutil

try:
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
except ImportError:
    raise ImportError("You must install the 'nltk' package to use this script."
                      "Please run 'pip install nltk'.")

from scilpy.utils.scilpy_bot import (
    _stem_keywords, update_matches_and_scores,
    _stem_phrase, _generate_help_files,
    _get_synonyms, _extract_keywords_and_phrases,
    _calculate_score, _make_title, prompt_user_for_object,
    _split_first_sentence, _highlight_keywords
)
from scilpy.utils.scilpy_bot import SPACING_LEN, VOCAB_FILE_PATH
from scilpy.io.utils import add_verbose_arg
from scilpy.version import version_string


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)

    p.add_argument('expressions', nargs='+',
                   help='Search the provided list of expressions.\n'
                        'Use quotes to search for phrases.')

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

    args.expressions = [expression.lower() for expression in args.expressions]

    # keywords are single words. Phrases are composed keywords
    keywords, phrases = _extract_keywords_and_phrases(args.expressions)

    with open(VOCAB_FILE_PATH, 'r') as f:
        vocab_data = json.load(f)

    # If synonyms are enabled, extend the search to include synonyms
    if not args.no_synonyms:
        all_expressions = keywords + phrases
        extended_expressions = set()
        for expression in all_expressions:
            synonyms = _get_synonyms(expression, vocab_data['synonyms'])
            extended_expressions.update(synonyms)
        extended_expressions.update(args.expressions)
        keywords, phrases = _extract_keywords_and_phrases(extended_expressions)

    stemmed_keywords = _stem_keywords(keywords)
    stemmed_phrases = list(set([_stem_phrase(phrase) for phrase in phrases]))

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

    scores_per_script = {}

    # Search through the docstrings of all scripts
    for script in sorted(hidden_dir.glob(f'scil_{selected_object}*.help')):
        script_name = script.stem

        with open(script, 'r') as f:
            search_text = f.read()

        score_details = _calculate_score(
            stemmed_keywords, stemmed_phrases, search_text,
            filename=script_name)

        scores_per_script = update_matches_and_scores(scores_per_script,
                                                      script_name, score_details)

    # Search in additional keywords in the vocabulary file
    for script in vocab_data['scripts']:
        if selected_object and selected_object not in script:
            continue

        script_name = script['name']
        script_keywords = script['keywords']
        search_text = ' '.join(script_keywords)
        score_details = _calculate_score(
            stemmed_keywords, stemmed_phrases, search_text, script_name)
        scores_per_script = update_matches_and_scores(scores_per_script,
                                                      script_name, score_details)

    # Remove scripts with no matches
    scores_per_script = {script: score for script,
                         score in scores_per_script.items() if score}
    matched_scripts = list(scores_per_script.keys())

    if not matched_scripts:
        logging.info(_make_title(' No results found! '))
        return

    total_scores = {match: sum(
        scores_per_script[match].values()) for match in matched_scripts}
    sorted_matches = sorted(total_scores, key=total_scores.get)

    # Sort matches by score and display them
    logging.info(_make_title(' Results Ordered by Score '))
    for match in sorted_matches:
        if total_scores[match] == 0:
            continue

        # Highlight keywords based on verbosity level
        with open(hidden_dir / f'{match}.help', 'r') as f:
            docstrings = f.read()

        all_experessions = stemmed_keywords + keywords + phrases \
            + stemmed_phrases
        if not args.no_synonyms:
            all_experessions += synonyms

        all_experessions = set(all_experessions)

        highlighted_docstring = _highlight_keywords(docstrings,
                                                    all_experessions)
        if args.verbose == 'INFO':
            first_sentence = _split_first_sentence(
                highlighted_docstring)[0]
            logging.info(f"{first_sentence.strip()}")
        elif args.verbose == 'DEBUG':
            logging.debug(f"{highlighted_docstring.strip()}")

        # Print the basic information at the end
        logging.info(
            f"{Fore.LIGHTYELLOW_EX}Total Score: {total_scores[match]}"
            f"{Style.RESET_ALL}")

        logging.info(
            f"{Fore.LIGHTBLUE_EX}{Style.BRIGHT}{match}{Style.RESET_ALL}")

        for word, score in scores_per_script[match].items():
            original_word = keyword_mapping.get(
                word, phrase_mapping.get(word, word))
            logging.info(
                f"{Fore.LIGHTGREEN_EX}Occurrence of '{original_word}': ' \
                f'{score}{Style.RESET_ALL}")
        logging.info(f"{Fore.LIGHTBLUE_EX}{'=' * SPACING_LEN}")
        logging.info("\n")

    logging.info(_make_title(
        ' Results Ordered by Score (Best results at the bottom) '))


if __name__ == '__main__':
    main()
