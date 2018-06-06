import pandas as pd
import numpy as np
import statsmodels.api as sm
import csv
from sklearn import linear_model
import json
import ast
import re

import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
from sklearn.linear_model import ElasticNet

from gensim import corpora
from gensim.models import Phrases
from gensim.models import Word2Vec
from gensim.models import Doc2Vec


def get_embedding(sentence, model):
	words = list(model.wv.vocab)
	# print(len(words))
	# X = model[model.wv.vocab]
	result = np.zeros(100)
	for word in sentence:
		result += model[word]
	if len(sentence) > 1:
		result /= len(sentence)
	return result

	# print model[sentences[0]]
	# print len(model[sentences[0]])
	# print model['opioid']
	# print len(model['opioid'])
	# print model['fighting']
	# print len(model['fighting'])


def write_matrix(matrix, filename):
	# write it
	table = matrix
	with open(filename, 'w') as csvfile:
		writer = csv.writer(csvfile)
		[writer.writerow(r) for r in table]

def read_matrix(filename):
# read it
	with open(filename, 'r') as csvfile:
		reader = csv.reader(csvfile)
		table = np.matrix([[float(e) for e in r] for r in reader])
		return table


def get_X():
	X = pd.read_csv('tweetsV2_' + str(1) + '.csv', sep='\t', encoding='utf-8', names=['FIPS', 'total_count', 'tweets'], skiprows=1)
	# f = open('tweetsV2_' + str(1) + '.csv', "r")
	# lines = f.readlines()
	# f.close()
	print X.columns
	for i in range(2, 10):
		print i
		X_temp = pd.read_csv('tweetsV2_' + str(i) + '.csv', sep='\t', encoding='utf-8', names=['FIPS', 'total_count', 'tweets'], skiprows=1)
		# f = open('tweetsV2_' + str(i) + '.csv', "r")
		# lines = f.readlines()
		# f.close()
		print X.columns
		X = pd.concat([X, X_temp], ignore_index=True)#, on ='FIPS', how='outer')
	print "++++++++++++++++X PR++++++++++++++++"
	print len(X['FIPS'].values.tolist())
	print X['FIPS'].values.tolist()
	X = X.fillna(value=0)
	#X = X.drop_duplicates()
	return X





def get_tweets(X, FIPS, single_doc=True):
	single_county_tweets = X[X.FIPS == FIPS].tweets.values[0]
	count_single_county_tweets = X[X.FIPS == FIPS].total_count.values[0]
	print count_single_county_tweets
	single_county_tweets = single_county_tweets.encode('utf=8', 'ignore')
	tweets = ""

	matches = []
	for match in re.finditer('{u\'lang\':', single_county_tweets):
	    print (match.start(), match.end())
	    matches.append((match.start(), match.end()))
	    #print single_county_tweets[match.start():match.end()+1]

	if count_single_county_tweets > 0:
		for i in range(count_single_county_tweets-1):
			single_tweet_dict = single_county_tweets[matches[i][0]:matches[i+1][0]-2]
			tweet_start = single_tweet_dict.find('text\': u\'')
			tweet_end = single_tweet_dict.find('\', u\'created_at\':')
			tweet = single_tweet_dict[tweet_start+9:tweet_end+1]
			#Remove links
			tweet = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', tweet)
			#Remove non alpha-numeric and lowercase all 
			tweet = re.sub(r'\W+', ' ', tweet).lower()
			tweets = tweets + tweet

		single_tweet_dict = single_county_tweets[matches[count_single_county_tweets-1][0]:-1]
		tweet_start = single_tweet_dict.find('text\': u\'')
		tweet_end = single_tweet_dict.find('\', u\'created_at\':')
		tweet = single_tweet_dict[tweet_start+9:tweet_end+1]
		#Remove links
		tweet = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', tweet)
		#Remove non alpha-numeric and lowercase all 
		tweet = re.sub(r'\W+', ' ', tweet).lower()
		tweets = tweets + tweet

	print tweets
	return tweets

#print json.loads(single_county_tweets, encoding='utf=8')
#print ast.literal_eval(single_county_tweets)
# ["A", "B", "C", " D"]
# print len(single_county_tweets)
# print list(single_county_tweets)
# print len(list(single_county_tweets))



X = get_X()
print X.shape
X = X.sort_values(by=['FIPS'])
#X = X.set_index('FIPS')
print X.columns

full_tweets = []
X_result = X.copy()
for i in range(608):
	fip = X.loc[i].FIPS#.values[0]
	if not isinstance(fip, int):
		fip = fip.values[0]
	print fip
	temp = get_tweets(X, fip)
	X_result.loc[i, 'tweets'] = temp
	full_tweets.append(temp.split())
#print X_result.tweets

# X_np = X.as_matrix()
# X_result_np = X.as_matrix()
# Y_np = read_matrix('Y_np')
# print X_np.shape
# print X_result_np.shape
# print Y_np.shape
#print X_result_np[:,2]

model = Word2Vec(full_tweets, min_count=1)#, # tokenized senteces, list of list of strings
        #          size=300,  # size of embedding vectors
        #          workers=4, # how many threads?
        #          min_count=20, # minimum frequency per token, filtering rare words
        #          sample=0.05, # weight of downsampling common words
        #          sg = 0, # should we use skip-gram? if 0, then cbow
        #          iter=5,
        #          hs = 0
        # )

X_embedded = []
#X.columns = ['FIPS', 'total_count',	'tweets_embedding']
for i in range(608):
	sentence = full_tweets[i]
	result_embedding = get_embedding(sentence, model)
	X_embedded.append(result_embedding)

print np.matrix(X_embedded)
X_embedded = np.matrix(X_embedded)
print X_embedded
#print X
dataframe = pd.DataFrame(data=X_embedded.astype(float))
dataframe.to_csv('X_embedded_np.csv', sep=' ', header=False, index=False)
#write_matrix(X_embedded, 'X_embedded_np')
# write_matrix(X_np, 'X_twitter_np')
# write_matrix(Y_np, 'Y_twitter_np')


