#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json 
import ast
from pathlib import Path



def _get_docstring_from_script_path(script_path):
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
    with open(script_path, 'r') as reader:
        file_contents = reader.read()
    module = ast.parse(file_contents)
    docstring = ast.get_docstring(module) or ''
    return docstring


def _get_help_text_from_file(help_file_path):
    with open(help_file_path, 'r') as f:
        help_text = f.read()
    return help_text


def generate_json(knowledge_base_dir, hidden_dir, output_json_dir):
    knowledge_base = {'scripts': []}

    for script in sorted(Path(knowledge_base_dir).glob('*.py')):
        script_name = script.stem
        if script_name in ('__init__','scil_search_keywords'):
            continue

        docstring = _get_docstring_from_script_path(str(script))
        help_file_path = Path(hidden_dir) / f'{script_name}.py.help'

        if not help_file_path.exists():
            print(f"Warning: Help file for {script_name} not found in {hidden_dir}")
            help_text = ''
        else:
            help_text = _get_help_text_from_file(help_file_path)
        
        script_info = {
            'name': script_name,
            'docstring': docstring,
            'help': help_text,
            'synonyms': [],  # This can be filled later by lab members
            'keywords': []   # This can be filled later by lab members
        }
        
        knowledge_base['scripts'].append(script_info)

    # Ensure the output directory exists
    output_json_dir.mkdir(parents=True, exist_ok=True)
    output_json_path = output_json_dir / 'knowledge_base.json'
    
    with open(output_json_path, 'w') as json_file:
        json.dump(knowledge_base, json_file, indent=4)
    
    print(f"Knowledge base JSON has been generated at {output_json_path}")


def main():
    base_dir = Path(__file__).parent.parent
    knowledge_base_dir = base_dir/'scripts/'
    hidden_dir = knowledge_base_dir / '.hidden'
    output_json_dir = base_dir/'scilpy-bot-scripts'/'json_files'

    generate_json(knowledge_base_dir, hidden_dir, output_json_dir)
    

if __name__ == '__main__':
    main()