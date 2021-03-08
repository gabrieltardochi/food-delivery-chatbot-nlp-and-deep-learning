import re
from unidecode import unidecode
import numpy as np
import nltk

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def fix_sentence(sentence):
    """
    keep only alphanumeric and space, and ignore non-ascii chars
    """
    return re.sub(r'[^A-Za-z0-9 ]+', '', unidecode(sentence)).strip().lower()


def tokenize(sentence):
    """
    split sentence into array of words/tokens
    a token can be a word or punctuation character, or number
    """
    return nltk.word_tokenize(sentence)


def stem(word):
    """
    stemming = find the root form of the word
    examples:
    words = ["organize", "organizes", "organizing"]
    words = [stem(w) for w in words]
    -> ["organ", "organ", "organ"]
    """
    return stemmer.stem(word)


def bag_of_words(tokenized_stemmed_sentence, words):
    """
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in tokenized_stemmed_sentence: 
            bag[idx] = 1
    return bag
