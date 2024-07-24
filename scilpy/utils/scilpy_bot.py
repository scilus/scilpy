
import re
import json
import ast
import nltk
import pathlib
import subprocess
from nltk.stem import PorterStemmer
from colorama import init, Fore, Style

stemmer = PorterStemmer()

RED = '\033[31m'
BOLD = '\033[1m'
END_COLOR = '\033[0m'
SPACING_CHAR = '='
SPACING_LEN = 80

# Path to the JSON file containing script information and keywords
KEYWORDS_FILE_PATH = pathlib.Path(__file__).parent /'Vocabulary'/'Keywords.json'
SYNONYMS_FILE_PATH = pathlib.Path(__file__).parent /'Vocabulary'/'Synonyms.json'




OBJECTS = [
    'aodf', 'bids', 'bingham', 'btensor', 'bundle', 'connectivity', 'denoising',
    'dki', 'dti','dwi', 'fodf', 'freewater', 'frf', 'gradients', 'header', 'json',
    'labels', 'lesions', 'mti', 'NODDI', 'sh', 'surface', 'tracking',
    'tractogram', 'viz', 'volume', 'qball', 'rgb', 'lesions'
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
            choice = int(input("Choose the object you want to work on (enter the number): "))
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
    return f'{Fore.BLUE}{Style.BRIGHT}{text.center(80, "=")}{Style.RESET_ALL}'


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
    return [stemmer.stem(keyword) for keyword in keywords]

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
    return ' '.join([stemmer.stem(word) for word in words])

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
    return ' '.join([stemmer.stem(word) for word in words])

def _generate_help_files():
    """
    Call the external script generate_help_files to generate help files
    """
    script_path = pathlib.Path(__file__).parent /'generate_help_files.py'
    #calling the extrernal script generate_help_files
    subprocess.run(['python', script_path], check=True)
    
def _highlight_keywords(text, stemmed_keywords):
    """
    Highlight the stemmed keywords in the given text using colorama.

    Parameters
    ----------
    text : str
        Text to highlight keywords in.
    stemmed_keywords : list of str
        Stemmed keywords to highlight.

    Returns
    -------
    str
        Text with highlighted keywords.
    """
    words = text.split()
    highlighted_text = []
    for word in words:
        stemmed_word = stemmer.stem(word)
        if stemmed_word in stemmed_keywords:
            highlighted_text.append(f'{Fore.RED}{Style.BRIGHT}{word}{Style.RESET_ALL}')
        else:
            highlighted_text.append(word)
    return ' '.join(highlighted_text)

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
    for synonym_set in synonyms_data['synonyms']:
        if keyword in synonym_set:
            return synonym_set
    return []

def _extract_keywords_and_phrases(keywords):
    """
    Extract keywords and phrases from the provided list.

    Parameters
    ----------
    keywords : list of str
        List of keywords and phrases.

    Returns
    -------
    list of str, list of str
        List of individual keywords and list of phrases.
    """
    keywords_list = []
    phrases_list = []

    for keyword in keywords:
        if ' ' in keyword: #if keyword contain blank space (contains more that 1 word)
            phrases_list.append(keyword)
        else:
            keywords_list.append(keyword)
    return keywords_list, phrases_list

def _calculate_score(keywords, phrases, text, filename):
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
        Score details based on the frequency of keywords in the text and filename.
    """
    stemmed_text = _stem_text(text.lower())
    stemmed_filename  = _stem_text(filename.lower())
    score_details = {'total_score': 0}

    for keyword in keywords:
        keyword = keyword.lower()
        keyword_score = stemmed_text.count(keyword) + stemmed_filename.count(keyword)
        score_details[keyword] = keyword_score
        score_details['total_score'] += keyword_score

    for phrase in phrases:
        phrase_stemmed = _stem_text(phrase.lower())
        phrase_score = stemmed_text.count(phrase_stemmed)
        score_details[phrase] = phrase_score
        score_details['total_score'] += phrase_score
    return score_details

