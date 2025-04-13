import re
import os
import numpy as np
import pandas as pd
import string
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet
nltk.download("punkt")
nltk.download("stopwords")
nltk.download('wordnet')
from nltk import pos_tag
nltk.download('averaged_perceptron_tagger_eng')
nltk.download("averaged_perceptron_tagger")
from nltk.util import ngrams
import spacy
import random
import string

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words("english"))



def clean_meta_data(dataset, test_data=True):
    dataset = dataset.copy()
    if "mmse" in dataset.columns:
        dataset.drop(columns=["mmse"], inplace=True)
    dataset.columns = [column.strip() for column in dataset.columns]
    for column in dataset.columns:
        dataset[column] = dataset[column].apply(lambda x:
            np.nan if pd.isna(x) else x.strip()
            if isinstance(x, str) else x)
    if test_data:
        dataset["gender"] = dataset["gender"].apply(lambda x: 1 if x=="male" else 0)
    dataset.index = dataset["ID"]
    dataset.drop(columns=["ID"], inplace=True)
    return dataset


def extract_speech_from_cha(file_path):
    """Extracts only the speech utterances from the .cha file."""
    speech_data = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if line.startswith("*PAR:"):
                clean_line = re.sub(r"\x15.*?\x15", "", line[5:]).strip()
                speech_data.append(clean_line)
    return speech_data


def extract_all_sentences(path):
    all_sentences = []
    for filename in os.listdir(path):
        if filename.endswith(".cha"):
            file_path = os.path.join(path, filename)
            speech_data = extract_speech_from_cha(file_path)
            all_sentences.append(speech_data)
    return all_sentences


# Initialize lemmatizer and stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def correct_spelling(sentence):
    """Corrects spelling using TextBlob."""
    return str(TextBlob(sentence).correct())

def lemmatize_word(word):
    pos_tagged = pos_tag([word])[0][1]
    if pos_tagged.startswith('NN'):  # Nouns
        return lemmatizer.lemmatize(word, pos='n')
    elif pos_tagged.startswith('VB'):  # Verbs
        return lemmatizer.lemmatize(word, pos='v')
    elif pos_tagged.startswith('JJ'):  # Adjectives
        return lemmatizer.lemmatize(word, pos='a')
    else:
        return word

def clean_text(sentence):
    # Tokenizes and cleans the sentence by removing stopwords, non-alphabetic words,
    # punctuation, and applies lemmatization and spelling correction.
    sentence = correct_spelling(sentence.lower())  # Correct spelling first
    tokens = word_tokenize(sentence)
    # Remove non-alphabetic words, stopwords, and punctuation
    filtered_tokens = [lemmatize_word(word) for word in tokens if word.isalpha() and word not in stop_words and word not in string.punctuation]
    return " ".join(filtered_tokens)







