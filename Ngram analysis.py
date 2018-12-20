#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 10:09:20 2018

@author: mbarugel
"""



import contractions as cont #only works with lowercase text
import utility_functions as util

import numpy as np
import pandas as pd

import math
import json

#%% Plotly
import plotly
plotly.tools.set_credentials_file(username='mbarugel', api_key='VeZxOOwfeVq5RJ34tEbe')
#%%

#%% Read data
print("loading data..")
full_traindata = pd.read_csv('/Users/mbarugel/Documents/Ngram Review analysis/tmp_reviews.gz')

#%% Select campaign
#traindata = pd.read_csv('/Users/mbarugel/Downloads/American Home Shield.csv')
#traindata = pd.read_csv('/Users/mbarugel/Downloads/CheapOAir.csv')
#traindata = pd.read_csv('/Users/mbarugel/Downloads/TruGreen.csv')
campaign_name =  'Comcast Internet Service'
traindata = full_traindata[(full_traindata['campaign_name'] == campaign_name) & \
                           full_traindata['original_rating'].notna() ]
y_binary = np.array(traindata['original_rating']) >= 4
y = np.array(traindata['original_rating']) 
x = list(traindata.moderated_text)
print('Working on',campaign_name,'with',str(traindata.shape[0]),'rows')
print('Avg Rating',str(round(np.mean(traindata['original_rating']),2)),
      'with std. dev.',str(round(np.std(traindata['original_rating']),2)))
print('Proportion of 4+ reviews',str(round(np.mean(y_binary), 2)))
#%% Preprocess text
x_2 = [item.lower() for item in x] #Lowercase
x_2 = [item.replace("’", "'") for item in x_2] #Correct apostrophe
x_2 = [cont.expandContractions(item) for item in x_2] #Expand contractions
x_2 = [item.replace("\r\n"," ") for item in x_2] # Remove line breaks
x_2 = [item.replace("american home shield", "companyname") for item in x_2]
x_2 = [item.replace("ahs", "companyname") for item in x_2]
x_2 = [item.replace("thank you very much", "") for item in x_2]
x_2 = [item.replace("thank you", "") for item in x_2]
x_2 = [item.replace("thanks", "") for item in x_2]
x_2 = [item.replace("would recommend", "") for item in x_2]
x_2 = [item.replace("not", "") for item in x_2]


#%% Analyze with continuous data
#X, feat_names = generate_term_matrix('all-ngrams', x_2)
#df = analyze_outcomes(X, y, feat_names, binary = False, include_regression=True, include_bivariate=True)

#%% Analyze binary with ngrams or keyphrases
X, feat_names = util.generate_term_matrix('keyphrases', x_2)
df = util.analyze_outcomes(X, y_binary, feat_names, binary = True, include_regression=True, include_bivariate=True)

#%% Analyze with nouns and related
X, feat_names = util.generate_term_matrix('nouns-and-related', x_2)
df = util.analyze_outcomes(X, y_binary, feat_names, binary = True, include_regression=True, include_bivariate=True)

print(df[(df['bad'] != True) & (df['pval'] <= 0.05)].sort_values('pval', ascending=True)[['name','odds_ratio', 'count']])
print(df[(df['bad'] != True) & (df['pval'] <= 0.05)].sort_values('count', ascending=False)[['name','odds_ratio', 'count']])

# Get related to any noun CheapoAir:
df[df['name'].str.startswith('ticket')][df['pval'] <= 0.05].sort_values('pval', ascending=True)[['name','odds_ratio', 'count']]
df[df['name'].str.startswith('agent')][df['pval'] <= 0.05].sort_values('pval', ascending=True)[['name','odds_ratio', 'count']]
df[df['name'].str.startswith('money')][df['pval'] <= 0.05].sort_values('pval', ascending=True)[['name','odds_ratio', 'count']]
df[df['name'].str.startswith('cancellation fee')][df['pval'] <= 0.05].sort_values('pval', ascending=True)[['name','odds_ratio', 'count']]

# Get related to any noun: AHS:
df[df['name'].str.startswith('claim')][df['pval'] <= 0.05].sort_values('count', ascending=False)[['name','odds_ratio', 'count']]
df[df['name'].str.startswith('contractor')][df['pval'] <= 0.05].sort_values('count', ascending=False)[['name','odds_ratio', 'count']]
df[df['name'].str.startswith('service')][df['pval'] <= 0.05].sort_values('count', ascending=False)[['name','odds_ratio', 'count']]

#%% Plot
df_to_plot = df[(df['bad'] != True) & (df['pval'] <= 0.05)]. \
  sort_values('pval', ascending=True)[['name','odds_ratio', 'count']][0:50]
df_to_plot['log_odds'] = [math.log(odds) for odds in df_to_plot['odds_ratio']]

a = dict(x=list(df_to_plot['count']),
    y=list(df_to_plot['log_odds']),
    text=list(df_to_plot['name']))
print(json.dumps(a))

#%% Get first 15 docs for each feature
n_features = X.shape[1]
doc_query = dict()
for i in range(n_features):
  if any(df_to_plot['name'] == feat_names[i]):
    doc_query[feat_names[i]] = [j for j, x in enumerate(X[:,i] != 0) if x ]
with open("doc_queries.js", "w") as f:
  print("doc_query_index = ", json.dumps(doc_query), "\n\n\n", file=f)
with open("doc_queries.js", "a") as f:
  print("docs = ", json.dumps(x), "\n\n\n", file=f)

#%%
