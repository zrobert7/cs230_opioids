#import pandas as pd
# import numpy as np
# import statsmodels.api as sm
# from sklearn import linear_model
import csv
# import matplotlib.pyplot as plt
# from sklearn.metrics import r2_score
# from sklearn.linear_model import ElasticNet
#import keras
import pandas as pd
import twitter 



api_key = '04pYTTtvGIlnz7TUkloRwiXFw'
api_secret = 'zmcAjPaC0mQ5M0gDYDOUhsOLOyoAHNq06emGz0RPAGF7z1h5Hi'
access_token = '4340540354-yDL10wiMNxkQd1ItyjEGC1VNyXSUSnYgGP4yezM'
access_token_secret = 'BcCkFTI76WReL33HIHfhqj03jKihrcH3Pk7DlaLVja8y1'
owner = 'zrobert77'


api = twitter.Api(consumer_key=api_key,
  consumer_secret=api_secret,
  access_token_key=access_token,
  access_token_secret=access_token_secret)

#print(api.VerifyCredentials())

def clean(x):
	x = str(x)[1:]
	x = int(x)
	return x

def get_county_db(filename):
	df = pd.read_csv(filename, delimiter='\t')
	print list(df.columns)
	df=df.rename(columns = {'INTPTLONG                                                                                                               ':'INTPTLONG'})
	print df[['GEOID','INTPTLAT','INTPTLONG']]
	# df['GEOID'] = df['GEOID'].apply(clean)
	# print df[['GEOID','INTPTLAT','INTPTLONG']]
	return df


def get_lat_lon_from_FIPS(fips, df):
	#TODO: also use land data to get county radius ! 
	row = df[df['GEOID'] == fips]
	print row
	return (float(row['INTPTLAT']), float(row['INTPTLONG']))

def get_related_terms():
	return ["opioid",
	"opioids", 
	'Fiorional', 
	'Codeine'
	'RobitussinA-C',
	'TylenolwithCodeine'
	'EmpirinwithCodeine',
	'Roxanol',
	'Duramorph',
	'Demerol',
	'CaptainCody',
	'Cody',
	'Schoolboy',
	'DoorsFours',
	'PancakesSyrup',
	'Loads',
	'M',
	'MissEmma',
	'Monkey',
	'WhiteStuff',
	'Demmies',
	'Painkiller',
	'Actiq',
	'Duragesic',
	'Sublimaze',
	'OxyContin',
	'Percodan',
	'Percocet',
	'Tylox',
	'Dilaudid',
	'Apache',
	'Chinagirl',
	'Dancefever',
	'Goodfella',
	'Murder8',
	'TangoandCash',
	'Chinawhite',
	'Friend',
	'Jackpot',
	'TNT',
	'Oxy80',
	'oxy',
	'Oxycat',
	'Hillbilly heroin',
	'Percs',
	'Perks',
	'Juice',
	'Dillies'
	 ]


def get_fips_list():
	#TODO get 608 county codes and prepend a zero if need be ?? or int()



def create_twitter_db(fips_codes, opioid_terms):
	county_db = get_county_db('data/2015_Gaz_counties_national.txt')
	for fips_code in fips_codes:
		(lat, lon) = get_lat_lon_from_FIPS(fips_code, county_db)
		geocode = str(res1[0]) + ',' + str(res1[1]) + ",20mi"
		county_tweet_count = 0
		county_tweets = []
		for term in opioid_terms:
			search = api.GetSearch(term, count=100, geocode=geocode)#until='2016-01-01') # Replace happy with your search
			county_tweet_count += len(search)
			county_tweets.append((tweet.location, tweet.text))


# county_db = get_county_db('data/2015_Gaz_counties_national.txt')
# res1 = get_lat_lon_from_FIPS(29051, county_db)
# print res1
# geocode = str(res1[0]) + ',' + str(res1[1]) + ",20mi"
# print geocode


search = api.GetSearch("OPIOID", count=100, geocode=geocode)#until='2016-01-01') # Replace happy with your search
print len(search)
tweets = [i.AsDict() for i in search]
for tweet in search:
	print(tweet.location, tweet.text)





