#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import spacy
from pathlib import Path

# Initialize SpaCy
nlp = spacy.load('en_core_web_md')

def load_knowledge_base(json_file):
    """Load the knowledge base from a JSON file."""
    with open(json_file, 'r') as f:
        knowledge_base = json.load(f)
    return knowledge_base

def contextual_search(query, knowledge_base, threshold=0.1):
    """Perform a contextual search based on the user query."""
    query_doc = nlp(query)
    best_match = None
    highest_similarity = 0

    for script in knowledge_base['scripts']:
        # Combine docstring, help text, synonyms, and keywords for better matching
        description = (
            script['docstring'] + ' ' + script['help'] + ' ' +
            ' '.join(script['synonyms']) + ' ' + ' '.join(script['keywords'])
        )
        description_doc = nlp(description)
        similarity = query_doc.similarity(description_doc)
        if similarity > highest_similarity and similarity>threshold:
            highest_similarity = similarity
            best_match = script

    return best_match, highest_similarity

def main():
    base_dir = Path(__file__).parent

    json_file = base_dir / 'json_files' / 'knowledge_base.json'
    
    # Load the knowledge base from JSON file
    knowledge_base = load_knowledge_base(json_file)
    
    # Example user query
    query = "I need a script that computes the SH coefficient directly on the raw DWI signal."

    
    # Perform contextual search
    best_match, similarity = contextual_search(query, knowledge_base)
    
    if best_match:
        print(f"The best match is {best_match['name']} with a similarity score of {similarity:.2f}")
        print(f"Docstring: {best_match['docstring']}")
        print(f"Help: {best_match['help']}")
    else:
        print("No relevant script found.")

if __name__ == '__main__':
    main()
