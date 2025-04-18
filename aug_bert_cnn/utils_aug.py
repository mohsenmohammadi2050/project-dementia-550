import nlpaug.augmenter.word as naw
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import random
import numpy as np


# Set all relevant seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Initialize Contextual Word Embeddings (CWE) augmenter using BERT model
cont_aug = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="substitute")

def contextual_augment(sentence):
    try:
        result = cont_aug.augment(sentence)
        return result if isinstance(result, list) else [result]
    except Exception as e:
        print(f"Error augmenting sentence: {e}")
        return []



# Augment and save as txt file
def augment_and_save_contextual_speech(cleaned_speech_groups, output_path, save=True):
    augmented_groups = [
        sum([contextual_augment(s) for s in group], []) 
        for group in cleaned_speech_groups
    ]
    if save:
        with open(output_path, 'w', encoding='utf-8') as f:
            for group in augmented_groups:
                joined = ", ".join(group)
                f.write(f"[{joined}]\n")

    return augmented_groups


# Cosine similarity to measure similarity between original and augmentated data
def cosine_sim(sentence1, sentence2):
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", analyzer='char').fit_transform([sentence1, sentence2])
    cosine_sim_matrix = cosine_similarity(vectorizer[0:1], vectorizer[1:2])
    return cosine_sim_matrix[0][0]

# Save sentences as a txt file
def save_grouped_sentences_to_file(sentence_groups, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for group in sentence_groups:
            joined = ", ".join(group)
            f.write(f"[{joined}]\n")
    
    print(f"file saved at: {output_path}")
