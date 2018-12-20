#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 10:09:20 2018

@author: mbarugel
"""

import scipy as sp

from nltk.stem import WordNetLemmatizer
from pywsd.utils import lemmatize_sentence
from rake_nltk import Rake
import nltk

import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm

import contractions as cont #only works with lowercase text


nlp = en_core_web_sm.load()

lemmatizer = WordNetLemmatizer()
# Uses stopwords for english from NLTK, and all puntuation characters by
# default
r = Rake()

text = """I bought a pre-owned vehicle and I'm not mechanically inclined. I just know enough to get in trouble, so I thought it was best to buy a warranty. I looked up Endurance and they offered the most coverage, so I bought them. Then, I have another vehicle so I bought a second coverage for it. The guy I talked to was friendly and good. Paying for the warranties was easy, too. They were able to charge my card and I paid upfront cash.

This past weekend, I went to the beach with my family and the 'check engine' light came on in one of my cars. So I took it to the shop. They went through the whole vehicle and quoted me for about $1,500 worth of work. I gave them my warranty information then the warranty came back and said that none of the things that I needed to fix in my vehicle was covered. That was a bit disappointing as I bought the coverage for that peace of mind knowing that if something goes on my vehicle, I'll be covered. So, I had to cancel the warranty. That way, I could get the money back and pay to fix my car."""

# Lemmatize sentence using pywsd
text2 = " ".join(lemmatize_sentence(text))

### Rake extract keyphrases
# Extraction given the text.
r.extract_keywords_from_text(text2)

# Extraction given the list of strings where each string is a sentence.
#r.extract_keywords_from_sentences(<list of sentences>)

# To get keyword phrases ranked highest to lowest.
print(r.get_ranked_phrases())

# To get keyword phrases ranked highest to lowest with scores.
r.get_ranked_phrases_with_scores()


# Get entities with spaCy
doc = nlp(text)
print([(X.text, X.label_) for X in doc.ents])


#%% TfIdf
import numpy as np
from sklearn import metrics,preprocessing,cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
import sklearn.linear_model as lm
import pandas as pd

# Read data
#traindata = pd.read_csv('/Users/mbarugel/Downloads/American Home Shield.csv')
#traindata = pd.read_csv('/Users/mbarugel/Downloads/CheapOAir.csv')
traindata = pd.read_csv('/Users/mbarugel/Downloads/TruGreen.csv')

print("loading data..")
y_binary = np.array(traindata['original_rating']) >= 4
y = np.array(traindata['original_rating']) 
x = list(traindata.moderated_text)

#%% Preprocess text
x_2 = [item.lower() for item in x] #Lowercase
x_2 = [item.replace("’", "'") for item in x_2] #Correct apostrophe
x_2 = [cont.expandContractions(item) for item in x_2] #Expand contractions
x_2 = [item.replace("\r\n"," ") for item in x_2] # Remove line breaks
x_2 = [item.replace("american home shield", "COMPANYNAME") for item in x_2]
x_2 = [item.replace("ahs", "COMPANYNAME") for item in x_2]
x_2 = [item.replace("thank you very much", "") for item in x_2]
x_2 = [item.replace("thank you", "") for item in x_2]
x_2 = [item.replace("thanks", "") for item in x_2]
x_2 = [item.replace("would recommend", "") for item in x_2]
x_2 = [item.replace("not", "") for item in x_2]

#%% Create terms matrices
def generate_term_matrix(type):
  tfv = TfidfVectorizer(min_df=10,  max_features=None, strip_accents='unicode',  
          analyzer='word',token_pattern=r'\w{1,}',ngram_range=(2, 4), 
          use_idf=False,binary=True, norm=None, stop_words=('the', 'a', 'and', 'to', 'this', 'your', 'an', 'for', 'i', 'was', 'very', 'thank', 'you','much'))
  print('fitting tf-idf')
  tfv.fit(x_2)
  X = tfv.transform(x_2)
  # Get feature names
  feat_names = tfv.get_feature_names()
  
#%% Analyze
def tfidf_outcomes(x_2, y, include_regression = True, include_bivariate = True):
  tfv = TfidfVectorizer(min_df=10,  max_features=None, strip_accents='unicode',  
          analyzer='word',token_pattern=r'\w{1,}',ngram_range=(2, 4), 
          use_idf=False,binary=True, norm=None, stop_words=('the', 'a', 'and', 'to', 'this', 'your', 'an', 'for', 'i', 'was', 'very', 'thank', 'you','much'))
  print('fitting tf-idf')
  tfv.fit(x_2)
  X = tfv.transform(x_2)
  # Get feature names
  feat_names = tfv.get_feature_names()
    
  # Regression
  if include_regression:
    
    rd = lm.LogisticRegression(penalty='l2', dual=True, tol=0.0001, 
                               C=1, fit_intercept=True, intercept_scaling=1.0, 
                               class_weight=None, random_state=None)
    
    print('running regression')
    print("20 Fold CV Score: ", np.mean(cross_validation.cross_val_score(rd, X, y, cv=20, scoring='roc_auc')))
    rd.fit(X,y)
    
    # Get coefs
    coefs = rd.coef_[0]
    df = pd.DataFrame({'name': feat_names, 'coefs': coefs, 'count': np.array(X.sum(0))[0], 'abscoefs': abs(coefs)})
    # Clean df of bad ngrams
    badkws = np.repeat(False, len(df))
    for i in range(0, len(df)):
      words = df['name'][i].split(' ')
      badkws[i] = sum([item in text.ENGLISH_STOP_WORDS for item in words]) == len(words)
    
    df['bad'] = badkws
    print('Regression results')
    print(df[df['bad'] != True].sort_values('abscoefs', ascending=False))
  
  # Bivariate Analysis
  if include_bivariate:
    pval = []
    odds = []
    for i in range(len(feat_names)):
      if i % 100 == 0:
        print("Bivariate Analysis " + str(i) + " of " + str(len(feat_names)))
      
      inds_yes = (X[:,i] == 1).toarray()[:,0]
      outcome_yes = np.mean(y[inds_yes])
      outcome_no = np.mean(y[~inds_yes])
      odds_ratio = (outcome_yes / (1.0000001 - outcome_yes)) /  \
        (outcome_no / (1.0000001 - outcome_no))
      obs = np.array([[sum(y[inds_yes]), len(y[inds_yes]) - sum(y[inds_yes])], \
                       [sum(y[~inds_yes]), len(y[~inds_yes]) - sum(y[~inds_yes])]])
      g, p, dof, expctd = sp.stats.chi2_contingency(obs, lambda_="log-likelihood")
      pval.append(p)
      odds.append(odds_ratio)
    
    df['odds_ratio'] = odds
    df['pval'] = pval
    print('Bivariate Analysis results')
    print(df[(df['bad'] != True) & (df['pval'] <= 0.05)].sort_values('pval', ascending=True)[['name','odds_ratio', 'count']])

  return 0

#%% 
  

#%% Play with keyphrases
#r.extract_keywords_from_text(x_2[998])
#print(r.get_ranked_phrases())


# Get vocabulary of keyphrases
all_phrases = []
for i in range(len(x_2)):
  r.extract_keywords_from_text(x_2[i])
  all_phrases = all_phrases + r.get_ranked_phrases()
all_phrases = list(set(all_phrases))

# Get phrases with 2+ words
all_phrases = [phrase for phrase in all_phrases if len(phrase.split(' ')) > 1]


data = []
indX = []
indY = []
for i in range(len(x_2)):
  if i % 100 == 0:
    print(i)
  r.extract_keywords_from_text(x_2[i])
  phrases_this_doc = r.get_ranked_phrases()
  for phrase in phrases_this_doc:
    try:
      ind = all_phrases.index(phrase)
      data.append(1)
      indX.append(i)
      indY.append(ind)
    except ValueError:
      pass
    
data = np.array(data)
indX = np.array(indX)    
indY = np.array(indY)    
Xphrase = sp.sparse.csr_matrix((data, (indX, indY)))

# Filter phrases with 10+ occurrences
ind_df = [i for i, x in enumerate(np.array(Xphrase.sum(0) >= 10)[0]) if x]
Xphrase = Xphrase[:, ind_df]
all_phrases = [all_phrases[ind] for ind in ind_df]

print("20 Fold CV Score: ", np.mean(cross_validation.cross_val_score(rd, Xphrase, y, cv=20, scoring='roc_auc')))

rd.fit(Xphrase,y)
coefs = rd.coef_[0]
df = pd.DataFrame({'name': all_phrases, 'coefs': coefs , 'abscoefs': abs(coefs), 'count': np.array(Xphrase.sum(0))[0]})
print(df.sort_values('abscoefs', ascending=False)[['name','coefs','count']])


#%% Bivariate Analysis
pval = []
odds = []
count = []
import math
for i in range(len(all_phrases)):
  if i % 100 == 0:
    print("Bivariate Analysis " + str(i) + " of " + str(len(all_phrases)))
  
  inds_yes = (Xphrase[:,i] == 1).toarray()[:,0]
  outcome_yes = np.mean(y[inds_yes])
  outcome_no = np.mean(y[~inds_yes])
  odds_ratio = (outcome_yes / (1.0000001 - outcome_yes)) /  \
    (outcome_no / (1.0000001 - outcome_no))
  obs = np.array([[sum(y[inds_yes]), len(y[inds_yes]) - sum(y[inds_yes])], \
                   [sum(y[~inds_yes]), len(y[~inds_yes]) - sum(y[~inds_yes])]])
  g, p, dof, expctd = sp.stats.chi2_contingency(obs, lambda_="log-likelihood")
  pval.append(p)
  odds.append(odds_ratio)
  count.append(np.sum(inds_yes))

odds2 = [round(min(odd_this, 999), 3) for odd_this in odds ]
df['odds_ratio'] = odds2
df['pval'] = pval
df['count'] = count

print(df[ (df['count'] >= 10) & (df['pval'] <= 0.05)].sort_values('pval', ascending=True)[['name','odds_ratio', 'count']])

#%% Entities (TO TRY: SPACY NOUN CHUNKS)
def get_entity_types(text):
  doc = nlp(text)
  return list(set([X.label_ for X in doc.ents]))

#all_entities = []
#for i in range(len(x_2)):
#  if i % 100 == 0:
#    print(i)
#  all_entities = all_entities + get_entity_types(x_2[i])
#all_entities = list(set(all_entities))
all_entities = ['TIME', 'MONEY', 'QUANTITY', 'DATE', 'PERSON', 'GPE', 'ORG']


data = []
indX = []
indY = []
for i in range(len(x_2)):
  if i % 100 == 0:
    print(i)
  phrases_this_doc = get_entity_types(x_2[i])
  for phrase in phrases_this_doc:
    try:
      ind = all_entities.index(phrase)
      data.append(1)
      indX.append(i)
      indY.append(ind)
    except ValueError:
      pass
    
data = np.array(data)
indX = np.array(indX)    
indY = np.array(indY)    
Xphrase = sp.sparse.csr_matrix((data, (indX, indY)))

print("20 Fold CV Score: ", np.mean(cross_validation.cross_val_score(rd, Xphrase, y, cv=20, scoring='roc_auc')))

rd.fit(Xphrase,y)
coefs = rd.coef_[0]
df = pd.DataFrame({'name': all_entities, 'coefs': coefs , 'abscoefs': abs(coefs)})
print(df.sort_values('abscoefs', ascending=False)[['name','coefs']])


pval = []
odds = []
count = []
import math
for i in range(len(all_entities)):
  if i % 100 == 0:
    print("Bivariate Analysis " + str(i) + " of " + str(len(all_entities)))
  
  inds_yes = (Xphrase[:,i] == 1).toarray()[:,0]
  outcome_yes = np.mean(y[inds_yes])
  outcome_no = np.mean(y[~inds_yes])
  odds_ratio = (outcome_yes / (1.0000001 - outcome_yes)) /  \
    (outcome_no / (1.0000001 - outcome_no))
  obs = np.array([[sum(y[inds_yes]), len(y[inds_yes]) - sum(y[inds_yes])], \
                   [sum(y[~inds_yes]), len(y[~inds_yes]) - sum(y[~inds_yes])]])
  g, p, dof, expctd = sp.stats.chi2_contingency(obs, lambda_="log-likelihood")
  pval.append(p)
  odds.append(odds_ratio)
  count.append(np.sum(inds_yes))

odds2 = [round(min(odd_this, 999), 3) for odd_this in odds ]
df['odds_ratio'] = odds2
df['pval'] = pval
df['count'] = count

print(df[df['count'] >= 10].sort_values('pval', ascending=True)[['name','odds_ratio', 'count']])
