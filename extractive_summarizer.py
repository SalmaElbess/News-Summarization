import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.porter import PorterStemmer
import math
import numpy as np

def apply_preprocessing(text):
    original_sentences = sent_tokenize(text.lower())
    sentences = []
    # Sentences Pre-processing
    for sent in original_sentences:
        tokens = [item for item in word_tokenize(sent) if item not in my_stopwords and item != '.' and item.isalpha()] # Preprocessing
        sentences.append(' '.join([stemmer.stem(token) for token in tokens])) # Stemming
    return original_sentences, sentences

def word_tf(word,sentence):
    #tf_score =  sentence.count(word)/ len(sentence)
    return sentence.count(word)/ len(sentence)

def word_idf(word, sentences):
    return math.log10(' '.join(sentences).count(word)/ len(sentences))

def word_in_sentence_tfidf(word, sentence, sentences):
    return word_tf(word,sentence)*word_idf(word, sentences)

def sentence_score(sentence, sentences):
    return sum([word_in_sentence_tfidf(word, sentence, sentences) for word in sentence])

def sentences_scores(sentences):
    return [sentence_score(sent, sentences) for sent in sentences]

def get_best_k_sentences_indecies(k, sentences, original_sentences):
    #lista = sorted(list(np.argsort(sentences_scores(sentences))[-k:-1]))
    return sorted(list(np.argsort(sentences_scores(sentences))[-k:]))

def get_best_k_sentences(k, sentences, original_sentences):
    return [original_sentences[idx] for idx in get_best_k_sentences_indecies(k, sentences, original_sentences)]

def summarize(text, k):
    # Apply preprocessing
    original_sentences, sentences = apply_preprocessing(text)
    # Summariez
    return ' '.join(get_best_k_sentences(k, sentences, original_sentences))
