#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 10:47:50 2018

@author: mbarugel
"""
import scipy as sp

from nltk.stem import WordNetLemmatizer
from pywsd.utils import lemmatize_sentence
from rake_nltk import Rake
import nltk

import en_core_web_sm
nlp = en_core_web_sm.load()

import spacy
from spacy.lemmatizer import Lemmatizer
from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES

from collections import Counter

from sklearn import metrics,preprocessing,cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
import sklearn.linear_model as lm

import numpy as np
import pandas as pd

from tqdm import tqdm

#%% Noun chunking
def get_nouns_and_related(input_text):
  lemmatizer = Lemmatizer(LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES)
  
  output = []
  doc = nlp(input_text)
  for chunk in doc.noun_chunks:
    #print(chunk.text, '-', chunk.root.text, '-', chunk.root.dep_, '-',
    #       chunk.root.head.text, '-', chunk.root.head.pos_)
    
    # Get nouns only in chunk
    nouns_only = []
    related_adjectives = []
    for l in chunk.subtree:
      if l.pos_ == "NOUN":
        lemmas = lemmatizer(l.text, 'NOUN')
        nouns_only.append(lemmas[0])
      elif l.pos_ == "ADJ":
        lemmas = lemmatizer(l.text, 'ADJ')
        related_adjectives.append(lemmas[0])
    chunk_nouns = [word for word in chunk.text.split(' ') if word in nouns_only]
    chunk_nouns = ' '.join(chunk_nouns).strip()
    
    # Get chunk with root
    chunk_with_root = ''
    chunk_with_root_clean = ''
    if chunk.root.dep_ == "nsubj":
      chunk_with_root = chunk.text + ' ' + chunk.root.head.text
      chunk_with_root_clean = chunk_nouns + ' ' + lemmatizer(chunk.root.head.text,chunk.root.head.pos_)[0]
    elif chunk.root.dep_ == "dobj":
      chunk_with_root = chunk.root.head.text + ' ' + chunk.text
      chunk_with_root_clean = lemmatizer(chunk.root.head.text,chunk.root.head.pos_)[0] + ' ' + chunk_nouns
    chunk_with_root_clean = chunk_with_root_clean.strip()
    # Related adjectives and adverbs from words related to head
    head = chunk.root.text
    children = [child.text for child in chunk.root.head.children]
    children_pos = [child.pos_ for child in chunk.root.head.children]
    siblings_dict = {}
    for i in range(len(children)):
      siblings_dict[children[i]] = children_pos[i]
    try:
      siblings_dict.pop(head)
    except:
      pass
    siblings_dict_small = {key: value for (key, value) in siblings_dict.items() if value == 'ADJ' or value == 'ADV'}
    related_words = list(siblings_dict_small) + related_adjectives
    # Related adjectives from the noun chunk
    
    
    #print('chunk and root:',chunk.text, '-', chunk_with_root)
    #print('chunk and root clean:',chunk_nouns, '-', chunk_with_root_clean)
    #print('related:',related_words)
    
    if len(chunk_with_root_clean.strip()) > 0:
      final_chunk_with_root = [chunk_nouns + '----' + chunk_with_root_clean]
    else:
      final_chunk_with_root = []
    all_words = [chunk_nouns] + final_chunk_with_root  \
      + [chunk_nouns+'----'+word.strip()   for word in related_words]
    #print('all_words:',all_words)
    #print('-----')
    if len(chunk_nouns) > 2:
      output = output + all_words
  
  return list(set(output))


#%% Create terms matrices
def generate_term_matrix(terms_type, x_2):
  
  if terms_type == 'all-ngrams':
    tfv = TfidfVectorizer(min_df=10,  max_features=None, strip_accents='unicode',  
            analyzer='word',token_pattern=r'\w{1,}',ngram_range=(2, 4), 
            use_idf=False,binary=True, norm=None, stop_words=('the', 'a', 'and', 'to', 'this', 'your', 'an', 'for', 'i', 'was', 'very', 'thank', 'you','much'))
    print('fitting tf-idf vectorizer')
    tfv.fit(x_2)
    X = tfv.transform(x_2)
    # Get feature names
    feat_names = tfv.get_feature_names()
  
  elif terms_type == 'keyphrases':
    # Get vocabulary of keyphrases
    print("Building all keyphraes")
    r = Rake()
    all_phrases = []
    doc_phrase_list = list() 
    for i in range(len(x_2)):
      r.extract_keywords_from_text(x_2[i])
      phrases_this_doc = r.get_ranked_phrases()
      all_phrases = all_phrases + phrases_this_doc
      doc_phrase_list.append(phrases_this_doc)
    all_phrases = list(set(all_phrases))
    
    # Get phrases with 2+ words
    all_phrases = [phrase for phrase in all_phrases if len(phrase.split(' ')) > 1]
    
    
    data = []
    indX = []
    indY = []
    for i in tqdm(range(len(x_2))):
      #if i % 200 == 0:
      #  print("Key phrases - processed " + str(i) + " of " + str(len(x_2)) + " docs", end="\r", flush=True)
      phrases_this_doc = doc_phrase_list[i]
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
    
    # Write to final variables
    X = Xphrase
    feat_names = all_phrases
    
  elif terms_type == 'nouns-and-related':
    # Get vocabulary of nouns
    print("Building all nouns and related")
    all_phrases = []
    doc_phrase_list = list()  
    for i in range(len(x_2)):
      phrases_this_doc = get_nouns_and_related(x_2[i])
      all_phrases = all_phrases + phrases_this_doc
      doc_phrase_list.append(phrases_this_doc)
      if i % 50 == 0:
        print("Building nouns and related - processed " + str(i) + " of " + str(len(x_2)) + " docs", end="\r", flush=True)
    
    all_phrases = list(set(all_phrases))
    
    data = []
    indX = []
    indY = []
    for i in range(len(x_2)):
      if i % 200 == 0:
        print("Nouns and related - processed " + str(i) + " of " + str(len(x_2)) + " docs", end="\r", flush=True)
      phrases_this_doc = doc_phrase_list[i]
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
    
    # Write to final variables
    X = Xphrase
    feat_names = all_phrases
  else:
    raise Exception('You need to specify the type of terms to use in the matrix: all-ngrams, keyphrases, nouns-and-related') 
    
  return X, feat_names
  
#%% Analyze
def analyze_outcomes(X, y, feat_names, binary = True, include_regression = True, include_bivariate = True):
  
  # Set up final dataframe
  df = pd.DataFrame({'name': feat_names, 'count': np.array(X.sum(0))[0]})
  # Clean df of bad ngrams
  badkws = np.repeat(False, len(df))
  for i in range(0, len(df)):
    words = df['name'][i].split(' ')
    badkws[i] = sum([item in text.ENGLISH_STOP_WORDS for item in words]) == len(words)
  
  df['bad'] = badkws
  
  # Regression
  if include_regression:
    if binary == True:
      rd = lm.LogisticRegression(penalty='l2', dual=True, tol=0.0001, 
                                 C=1, fit_intercept=True, intercept_scaling=1.0, 
                                 class_weight=None, random_state=None)
      
      print('running regression')
      print("20 Fold CV Score: ", np.mean(cross_validation.cross_val_score(rd, X, y, cv=20, scoring='roc_auc')))
    elif binary == False:
      print('running regression')
      alpha_values_to_test = [2**i for i in range(-6,7)]
      best_cv_score = -9999
      for alpha_value in alpha_values_to_test:
        rd = lm.Ridge(alpha=alpha_value, fit_intercept=True, normalize = False)
        cv_score = np.mean(cross_validation.cross_val_score(rd, X, y, cv=20, scoring='r2'))
        if cv_score > best_cv_score:
          best_cv_score = cv_score
          best_alpha = alpha_value
          
      rd = lm.Ridge(alpha=best_alpha, fit_intercept=True, normalize = False)
      print("Best 20-Fold CV Score: ", np.mean(cross_validation.cross_val_score(rd, X, y, cv=20, scoring='r2')))
      
    rd.fit(X,y)
    
    # Get coefs
    if binary == True:
      coefs = rd.coef_[0]
    else:
      coefs = rd.coef_
    
    df['coefs'] = coefs
    df['abscoefs'] = abs(coefs)
    
    print('Regression results')
    print(df[df['bad'] != True].sort_values('abscoefs', ascending=False)[['name','coefs', 'count']])
  
  # Bivariate Analysis
  if include_bivariate and binary == True:
    pval = []
    odds = []
    for i in range(len(feat_names)):
      if i % 100 == 0:
        print("Bivariate Analysis " + str(i) + " of " + str(len(feat_names)), end="\r", flush=True)
      
      inds_yes = (X[:,i] == 1).toarray()[:,0]
      outcome_yes = np.mean(y[inds_yes])
      outcome_no = np.mean(y[~inds_yes])
      odds_ratio = (outcome_yes / (1.0000001 - outcome_yes)) /  \
        (outcome_no / (1.0000001 - outcome_no))
      obs = np.array([[sum(y[inds_yes]), len(y[inds_yes]) - sum(y[inds_yes])], \
                       [sum(y[~inds_yes]), len(y[~inds_yes]) - sum(y[~inds_yes])]])
      g, p, dof, expctd = sp.stats.chi2_contingency(obs, lambda_="log-likelihood")
      pval.append(p)
      odds.append(max(min(odds_ratio, 100), 0.01))
    
    df['odds_ratio'] = odds
    df['pval'] = pval
    print('Bivariate Analysis results')
    print(df[(df['bad'] != True) & (df['pval'] <= 0.05)].sort_values('pval', ascending=True)[['name','odds_ratio', 'count']])
    
  elif include_bivariate and binary == False:
    pval = []
    diffs = []
    for i in range(len(feat_names)):
      if i % 100 == 0:
        print("Bivariate Analysis " + str(i) + " of " + str(len(feat_names)))
      inds_yes = (X[:,i] == 1).toarray()[:,0]
      diff = np.mean(y[inds_yes]) - np.mean(y[~inds_yes])
      p = sp.stats.ttest_ind(y[inds_yes],y[~inds_yes], equal_var=False).pvalue
      pval.append(p)
      diffs.append(diff)
    
    df['diffs'] = diffs
    df['pval'] = pval
    print('Bivariate Analysis results')
    print(df[(df['bad'] != True) & (df['pval'] <= 0.05)].sort_values('pval', ascending=True)[['name','diffs', 'count']])
  return df
