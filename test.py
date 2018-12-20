#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 17:45:46 2018

@author: mbarugel
"""
import nltk 
from sklearn.feature_extraction import text as tt

text = """I bought a pre-owned vehicle and I'm not mechanically inclined. I just know enough to get in trouble, so I thought it was best to buy a warranty. I looked up Endurance and they offered the most coverage, so I bought them. Then, I have another vehicle so I bought a second coverage for it. The guy I talked to was friendly and good. Paying for the warranties was easy, too. They were able to charge my card and I paid upfront cash.

This past weekend, I went to the beach with my family and the 'check engine' light came on in one of my cars. So I took it to the shop. They went through the whole vehicle and quoted me for about $1,500 worth of work. I gave them my warranty information then the warranty came back and said that none of the things that I needed to fix in my vehicle was covered. That was a bit disappointing as I bought the coverage for that peace of mind knowing that if something goes on my vehicle, I'll be covered. So, I had to cancel the warranty. That way, I could get the money back and pay to fix my car."""
print(text)

sentence_re = r'(?:(?:[A-Z])(?:.[A-Z])+.?)|(?:\w+(?:-\w+)*)|(?:\$?\d+(?:.\d+)?%?)|(?:...|)(?:[][.,;"\'?():-_`])'
lemmatizer = nltk.WordNetLemmatizer()
stemmer = nltk.stem.porter.PorterStemmer()
grammar = r"""
    NBAR:
        {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns
        
    NP:
        {<NBAR>}
        {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
"""
chunker = nltk.RegexpParser(grammar)
toks = nltk.regexp_tokenize(text, sentence_re)
postoks = nltk.tag.pos_tag(toks)
print(postoks)
tree = chunker.parse(postoks)
stopwords = tt.ENGLISH_STOP_WORDS

def leaves(tree):
    """Finds NP (nounphrase) leaf nodes of a chunk tree."""
    for subtree in tree.subtrees(filter = lambda t: t.label()=='NP'):
        yield subtree.leaves()

def normalise(word):
    """Normalises words to lowercase and stems and lemmatizes it."""
    word = word.lower()
    # word = stemmer.stem_word(word) #if we consider stemmer then results comes with stemmed word, but in this case word will not match with comment
    word = lemmatizer.lemmatize(word)
    return word

def acceptable_word(word):
    """Checks conditions for acceptable word: length, stopword. We can increase the length if we want to consider large phrase"""
    accepted = bool(2 <= len(word) <= 40
        and word.lower() not in stopwords)
    return accepted


def get_terms(tree):
    for leaf in leaves(tree):
        term = [ normalise(w) for w,t in leaf if acceptable_word(w) ]
        yield term

terms = get_terms(tree)

# Print terms
print('printing terms')
for term in terms:
  print(term)

print('printing words')
for term in terms:
    for word in term:
        print (word)