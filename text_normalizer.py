import re
import nltk
import spacy
import unicodedata

from bs4 import BeautifulSoup
from contractions import CONTRACTION_MAP
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.tokenize import TweetTokenizer


tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
nlp = spacy.load('en_core_web_sm')


def remove_html_tags(text):
    text = BeautifulSoup(text).text
    return text


def stem_text(text):
    # HOLA
    return text


def lemmatize_text(text):
    # Put your code
    return text


def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    expanded_words=[]
    is_first_word = True

    for w in TweetTokenizer().tokenize(text):
        if w in contraction_mapping:
            w = contraction_mapping[w]
            if not is_first_word:
                w = " " + w
        else:
            if not is_first_word:
                if w.isalpha():
                    w = " " + w

        is_first_word = False
        expanded_words.append(w)
        
    text = "".join(expanded_words)
    return text


def remove_accented_chars(text):
    text = unicodedata.normalize("NFD", text).encode('ascii', 'ignore').decode("utf-8")
    return text


def remove_special_chars(text, remove_digits=False):
    # Put your code
    return text


def remove_stopwords(text, is_lower_case=False, stopwords=stopword_list):
    # Put your code
    return text


def remove_extra_new_lines(text):
    text = text.replace("\n", "")
    return text


def remove_extra_whitespace(text):
    # HOLOA
    return text
    

def normalize_corpus(
    corpus,
    html_stripping=True,
    contraction_expansion=True,
    accented_char_removal=True,
    text_lower_case=True,
    text_stemming=False,
    text_lemmatization=False,
    special_char_removal=True,
    remove_digits=True,
    stopword_removal=True,
    stopwords=stopword_list
):
    
    normalized_corpus = []

    # Normalize each doc in the corpus
    for doc in corpus:
        # Remove HTML
        if html_stripping:
            doc = remove_html_tags(doc)
            
        # Remove extra newlines
        doc = remove_extra_new_lines(doc)
        
        # Remove accented chars
        if accented_char_removal:
            doc = remove_accented_chars(doc)
            
        # Expand contractions    
        if contraction_expansion:
            doc = expand_contractions(doc)
            
        # Lemmatize text
        if text_lemmatization:
            doc = lemmatize_text(doc)
            
        # Stemming text
        if text_stemming and not text_lemmatization:
            doc = stem_text(doc)
            
        # Remove special chars and\or digits    
        if special_char_removal:
            doc = remove_special_chars(
                doc,
                remove_digits=remove_digits
            )  

        # Remove extra whitespace
        doc = remove_extra_whitespace(doc)

         # Lowercase the text    
        if text_lower_case:
            doc = doc.lower()

        # Remove stopwords
        if stopword_removal:
            doc = remove_stopwords(
                doc,
                is_lower_case=text_lower_case,
                stopwords=stopwords
            )

        # Remove extra whitespace
        doc = remove_extra_whitespace(doc)
        doc = doc.strip()
            
        normalized_corpus.append(doc)
        
    return normalized_corpus
