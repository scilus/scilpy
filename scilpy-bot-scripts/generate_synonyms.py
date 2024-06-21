import json
import gensim.downloader as api
from scipy.spatial.distance import cosine
import re 
from pathlib import Path

#Load vocabulary and Acronyms
def load_vocabulary(vocab_file_path):
    with open(vocab_file_path, 'r', encoding='utf-8') as file:
        vocabulary = [line.strip() for line in file]
    return vocabulary

def load_acronyms(acronyms_file_path):
    with open(acronyms_file_path, 'r', encoding='utf-8') as file:
        acronyms = json.load(file)
    return {entry['abbreviation']: entry['Description'] for entry in acronyms}

#load pre-trained word vectors
word_vectors = api.load("word2vec-google-news-300")

#calculate similarity and find synonyms
def get_word_embedding(word):
    if word in word_vectors:
        return word_vectors[word]
    return None

def calculate_similarity(word1, word2):
    embedding1 = get_word_embedding(word1)
    embedding2 = get_word_embedding(word2)
    if embedding1 is not None and embedding2 is not None:
        return 1 - cosine(embedding1, embedding2)
    return 0

def find_synonyms(word, vocabulary, acronyms_dict, threshold=0.7):
    synonyms = []
    for vocab_word in vocabulary:
        # Check if it's an acronym
        if vocab_word.startswith('*'):
            acronym = vocab_word[1:]
            if acronym in acronyms_dict:
                description = acronyms_dict[acronym]
                description_words = description.split()
                for desc_word in description_words:
                    similarity = calculate_similarity(word, desc_word)
                    if similarity >= threshold:
                        synonyms.append(vocab_word)
                        break 
        else:
            similarity = calculate_similarity(word, vocab_word)
            if similarity >= threshold:
                synonyms.append(vocab_word)
    return synonyms

def extract_words(text):
    return re.findall(r'\w+', text.lower())


def generate_synonyms(script_entry, vocabulary, acronyms_dict):
    words = set(extract_words(script_entry["docstring"]) + extract_words(script_entry["help"]))
    synonyms_dict = {}
    for word in words:
        synonyms = find_synonyms(word, vocabulary, acronyms_dict)
        if len(synonyms) != 0:
            synonyms.append(word)
            #synonyms_dict[word] = synonyms
            script_entry['synonyms'].append(synonyms)
    return script_entry

def update_scripts_with_synonyms(json_filepath, vocabulary, acronyms_dict):
    with open(json_filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    for script_entry in data['scripts']:
        script_entry['synonyms'] = []  # Initialize the synonyms list
        updated_script = generate_synonyms(script_entry, vocabulary, acronyms_dict)
    
    with open(json_filepath, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)


base_dir = Path(__file__).parent

vocab_filepath = base_dir/'json_files'/'Scilpy_vocabulary.txt'
acronyms_filepath = base_dir/'json_files'/'acronyms.json'
json_filepath = base_dir/'json_files'/'knowledge_base_word2vec.json'
vocabulary = load_vocabulary(vocab_filepath)
acronyms_dict = load_acronyms(acronyms_filepath)

update_scripts_with_synonyms(json_filepath, vocabulary, acronyms_dict)
print(f"Scripts in {json_filepath} have been updated with synonyms.")
