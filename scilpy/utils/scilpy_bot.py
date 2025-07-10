# -*- coding: utf-8 -*-
import ast
from colorama import Fore, Style
import pathlib
import re
import subprocess

import nltk
from nltk.stem import PorterStemmer
from tqdm import tqdm

SPACING_LEN = 80

stemmer = PorterStemmer()
try:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
except ImportError:
    raise ImportError("You must install the 'nltk' package to use this script."
                      "Please run 'pip install nltk'.")

# Path to the JSON file containing script information and keywords
VOCAB_FILE_PATH = pathlib.Path(
    __file__).parent.parent.parent/'data' / 'vocabulary.json'


OBJECTS = [
    'aodf', 'bids', 'bingham', 'btensor', 'bundle',
    'connectivity', 'denoising', 'dki', 'dti', 'dwi',
    'fodf', 'freewater', 'frf', 'gradients', 'header',
    'json', 'labels', 'lesions', 'mti', 'NODDI', 'sh',
    'surface', 'tracking', 'tractogram', 'viz', 'volume',
    'qball', 'rgb', 'lesions'
]


def prompt_user_for_object():
    """
    Prompts the user to select an object from the list of available objects.
    """
    print("Available objects:")
    for idx, obj in enumerate(OBJECTS):
        print(f"{idx + 1}. {obj}")
    while True:
        try:
            choice = int(
                input("Choose the object you want to work on "
                      "(enter the number): "))
            if 1 <= choice <= len(OBJECTS):
                return OBJECTS[choice - 1]
            else:
                print(f"Please enter a number between 1 and {len(OBJECTS)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def _make_title(text):
    """
    Returns a formatted title string with centered text and spacing
    """
    return f'{Fore.LIGHTBLUE_EX}{Style.BRIGHT}{text.center(SPACING_LEN, "=")}' \
           f'{Style.RESET_ALL}'


def _get_docstring_from_script_path(script):
    """
    Extract a python file's docstring from a filepath.

    Parameters
    ----------
    script : str
        Path to python file

    Returns
    -------
    docstring : str
        The file's docstring, or an empty string if there was no docstring.
    """
    with open(script, 'r') as reader:
        file_contents = reader.read()
    module = ast.parse(file_contents)
    docstring = ast.get_docstring(module) or ''
    return docstring


def _split_first_sentence(text):
    """
    Split the first sentence from the rest of a string by finding the first
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


def _stem_word(word):
    """
    Stem a word using two different stemmers and return the most appropriate
    stem.

    Parameters
    ----------
    word : str
        Word to stem.

    Returns
    -------
    str
        Stemmed word.
    """
    if len(word) <= 3:
        return word
    version_b = stemmer.stem(word)
    return version_b


def _stem_keywords(keywords):
    """
    Stem a list of keywords using PorterStemmer.

    Parameters
    ----------
    keywords : list of str
        Keywords to be stemmed.

    Returns
    -------
    list of str
        Stemmed keywords.
    """
    return [_stem_word(keyword) for keyword in keywords]


def _stem_text(text):
    """
    Stem all words in a text using PorterStemmer.

    Parameters
    ----------
    text : str
        Text to be stemmed.

    Returns
    -------
    str
        Stemmed text.
    """
    words = nltk.word_tokenize(text)
    return ' '.join([_stem_word(word) for word in words])


def _stem_phrase(phrase):
    """
    Stem all words in a phrase using PorterStemmer.

    Parameters
    ----------
    phrase : str
        Phrase to be stemmed.

    Returns
    -------
    str
        Stemmed phrase.
    """
    words = phrase.split()
    return ' '.join([_stem_word(word) for word in words])


def _generate_help_files():
    """
    This function iterates over all Python scripts in the 'scripts' directory,
    runs each script with the '--h' flag to generate help text,
    and saves the output in the '.hidden' directory.

    By doing this, we can precompute the help outputs for each script,
    which can be useful for faster searches.

    If a help file already exists for a script, the script is skipped,
    and the existing help file is left unchanged.

    The help output is saved in a hidden directory to avoid clutter in
    the main scripts directory.
    """

    scripts_dir = pathlib.Path(__file__).parent.parent.parent / 'scripts'
    help_dir = scripts_dir / '.hidden'

    scripts = [script for script in scripts_dir.glob('*.py')
               if script.name not in ['__init__.py',
                                      'scil_search_keywords.py']]

    helps = [help for help in help_dir.glob('*.help')]
    scripts_to_regenerate = [script for script in scripts
                             if help_dir / f'{script.name}.help' not in helps]

    # Check if all help files are present
    if len(scripts_to_regenerate) == 0:
        print("All help files are already generated.")
        return

    # Hidden directory to store help files
    hidden_dir = scripts_dir / '.hidden'
    hidden_dir.mkdir(exist_ok=True)

    # Iterate over all scripts and generate help files
    for script in tqdm(scripts_to_regenerate):
        help_file = hidden_dir / f'{script.name}.help'
        # Check if help file already exists
        if help_file.exists():
            continue

        # Run the script with --h and capture the output
        result = subprocess.run(['python', script, '--h'],
                                capture_output=True, text=True)

        # Save the output to the hidden file
        with open(help_file, 'w') as f:
            f.write(result.stdout)


def _highlight_keywords(text, all_expressions):
    """
    Highlight the stemmed keywords in the given text using colorama.

    Parameters
    ----------
    text : str
        Text to highlight keywords in.
    all_expressions : list of str
        List of all things to highlight.

    Returns
    -------
    str
        Text with highlighted keywords.
    """
    # Iterate over each keyword in the list
    for kw in all_expressions:
        # Create a regex pattern to match any word containing the keyword
        pattern = re.compile(
            r'\b(\w?' + re.escape(kw) + r's?\w?)\b', re.IGNORECASE)

        # Function to apply highlighting to the matched word
        def apply_highlight(match):
            return f'{Fore.LIGHTYELLOW_EX}{Style.BRIGHT}{match.group(0)}' \
                   f'{Style.RESET_ALL}'

        # Replace the matched word with its highlighted version
        text = pattern.sub(apply_highlight, text)

    return text


def _get_synonyms(keyword, synonyms_data):
    """
    Get synonyms for a given keyword from the synonyms data.

    Parameters
    ----------
    keyword : str
        Keyword to find synonyms for.
    synonyms_data : dict
        Dictionary containing synonyms data.

    Returns
    -------
    list of str
        List of synonyms for the given keyword.
    """
    keyword = keyword.lower()
    complete_synonyms = []
    for synonym_set in synonyms_data:
        synonym_set = [synonym.lower() for synonym in synonym_set]
        stemmed_synonyms_set = [_stem_word(synonym) for synonym in synonym_set]

        if keyword in synonym_set or _stem_word(keyword) in stemmed_synonyms_set:
            complete_synonyms.extend(synonym_set)

    return list(set(complete_synonyms))


def _extract_keywords_and_phrases(expressions):
    """
    Extract keywords and phrases from the provided list.

    Parameters
    ----------
    expressions : list of str
        List of keywords and phrases.

    Returns
    -------
    list of str, list of str
        List of individual keywords and list of phrases.
    """
    keywords_set = set()
    phrases_set = set()

    for expression in expressions:
        # if keyword contain blank space (contains more that 1 word)
        if ' ' in expression:
            phrases_set.add(expression.lower())
        else:
            keywords_set.add(expression.lower())

    return list(keywords_set), list(phrases_set)


def _calculate_score(keywords, phrases, text, filename, suffix=''):
    """
    Calculate a score for how well the text and filename match the keywords.

    Parameters
    ----------
    keywords : list of str
        Keywords to search for.
    phrases : list of str
        Phrases to search for.
    text : str
        Text to search within.
    filename : str
        Filename to search within.

    Returns
    -------
    dict
        Score details based on the frequency of keywords
        in the text and filename.
    """
    stemmed_text = _stem_text(text.lower())
    stemmed_filename = _stem_text(filename.lower())
    score_details = {}

    def is_match(found_word, keyword):
        if len(keyword) <= 3:
            return found_word == keyword
        return _stem_word(found_word) == _stem_word(keyword)

    for keyword in keywords:
        keyword = keyword.lower()
        # Use regular expressions to match whole words only

        keyword_pattern = re.compile(
            r'\b(\w?' + re.escape(keyword) + r's?\w?)\b', re.IGNORECASE)
        found_words = keyword_pattern.findall(stemmed_text) \
            + keyword_pattern.findall(stemmed_filename)
        keyword_score = 0

        for found_word in found_words:
            if is_match(found_word, keyword):
                keyword_score += 1
                continue

        if keyword_score > 0:
            score_details[keyword + suffix] = keyword_score

    for phrase in phrases:
        phrase_stemmed = _stem_text(phrase.lower())
        phrase_score = stemmed_text.count(phrase_stemmed)
        if phrase_score > 0:
            score_details[phrase + suffix] = phrase_score

    return score_details


def update_matches_and_scores(scores, filename, score_details):
    """
    Update the matches and scores for the given filename based
    on the score details.

    Parameters
    ----------
    scores : dict
        A dictionary containing the scores for the keywords (to be updated).
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

    return scores
