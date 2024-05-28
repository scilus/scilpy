import argparse
import logging
import pathlib
import re
import subprocess
import ast

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
    hidden_dir = script_dir / '.hidden'
    matches = []

    keywords_regexes = [re.compile('(' + re.escape(kw) + ')', re.IGNORECASE)
                        for kw in args.keywords]

    if args.search_parser:
        # Use precomputed help files
        for help_file in sorted(hidden_dir.glob('*.help')):
            script_name = help_file.stem
            with open(help_file, 'r') as f:
                search_text = f.read()

            # Test intersection of all keywords in the help text
            if not _test_matching_keywords(args.keywords, [script_name, search_text]):
                continue

            matches.append(script_name)
            search_text = search_text or 'No help text available!'

            display_filename = script_name
            display_short_info, display_long_info = _split_first_sentence(search_text)

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

            header = header.replace(END_COLOR, END_COLOR + BOLD) + END_COLOR
            footer = footer.replace(END_COLOR, END_COLOR + BOLD) + END_COLOR

            # Print everything
            logging.info(header)
            logging.info(display_short_info)
            logging.debug(display_long_info)
            logging.info(footer)
            logging.info("\n")
    else:
        # Extract docstrings directly from scripts
        for script in sorted(script_dir.glob('*.py')):
            filename = script.name
            if filename == '__init__.py' or filename == 'generate_help_files.py':
                continue

            search_text = _get_docstring_from_script_path(str(script))

            # Test intersection of all keywords in the docstring
            if not _test_matching_keywords(args.keywords, [filename, search_text]):
                continue

            matches.append(filename)
            search_text = search_text or 'No docstring available!'

            display_filename = filename
            display_short_info, display_long_info = _split_first_sentence(search_text)

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

            header = header.replace(END_COLOR, END_COLOR + BOLD) + END_COLOR
            footer = footer.replace(END_COLOR, END_COLOR + BOLD) + END_COLOR

            # Print everything
            logging.info(header)
            logging.info(display_short_info)
            logging.debug(display_long_info)
            logging.info(footer)
            logging.info("\n")

    if not matches:
        logging.info(_make_title(' No results found! '))

def _make_title(text):
    return BOLD + text.center(SPACING_LEN, SPACING_CHAR) + END_COLOR

def _test_matching_keywords(keywords, texts):
    matches = []
    for key in keywords:
        key_match = False
        for text in texts:
            if key.lower() in text.lower():
                key_match = True
                break
        matches.append(key_match)
    return all(matches)

def _get_docstring_from_script_path(script):
    with open(script, 'r') as reader:
        file_contents = reader.read()
    module = ast.parse(file_contents)
    docstring = ast.get_docstring(module) or ''
    return docstring

def _split_first_sentence(text):
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

