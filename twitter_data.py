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

#print(api.VerifyCredentials())

def clean(x):
	x = str(x)[1:]
	x = int(x)
	return x

def get_county_db(filename):
	df = pd.read_csv(filename, delimiter='\t')
	#print list(df.columns)
	df=df.rename(columns = {'INTPTLONG                                                                                                               ':'INTPTLONG'})
	#print df[['GEOID','INTPTLAT','INTPTLONG']]
	# df['GEOID'] = df['GEOID'].apply(clean)
	# print df[['GEOID','INTPTLAT','INTPTLONG']]
	return df


def get_lat_lon_from_FIPS(fips, df):
	#TODO: also use land data to get county radius ! 
	row = df[df['GEOID'] == fips]
	#print row
	return (float(row['INTPTLAT']), float(row['INTPTLONG']))

def get_related_terms():
	return ["opioid",
	"opioids", 
	'Fiorional', 
	'Fentanyl',
	'Codeine'
	'RobitussinA-C',
	'Robitussin',
	'TylenolwithCodeine'
	'EmpirinwithCodeine',
	'Roxanol',
	'Duramorph',
	'Demerol',
	'CaptainCody',
	#'Cody',
	#'Schoolboy',
	'DoorsFours',
	'PancakesSyrup',
	#'Loads',
	#'M',
	'MissEmma',
	#'Monkey',
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
	#'Apache',
	'Chinagirl',
	'Dancefever',
	'Goodfella',
	'Murder8',
	'TangoandCash',
	'Chinawhite',
	'MexicanBrown'
	#'Friend',
	#'Jackpot',
	#'TNT',
	'Oxy80',
	'oxy',
	'Oxycat',
	'oxycodone',
	'Vicodin',
	'hydrocodone',
	'Hillbillyheroin',
	'heroin',
	'Percs',
	'Perks',
	#'Juice',
	'superice',
	'Dillies'
	 ]


def get_fips_list():
	#TODO get 608 county codes and prepend a zero if need be ?? or int()
	df_codes = pd.read_csv('codes_x_np', delimiter='\n', header=None)
	codes_x_np = df_codes.as_matrix()
	result = codes_x_np.reshape((608,))
	return result

	



def create_twitter_db(api, fips_codes, opioid_terms):
	county_db = get_county_db('data/2015_Gaz_counties_national.txt')
	counts = []
	tweets = []
	codes = []
	after_num = False
	for fips_code in fips_codes:
		if int(fips_code) == 48439:
			after_num = True
		if after_num:
			codes.append(fips_code)
			print "Getting data for county #" + str(fips_code)
			(lat, lon) = get_lat_lon_from_FIPS(fips_code, county_db)
			#"latitude,longitude,radius"
			geocode = str(lat) + ',' + str(lon) + ",20mi"
			county_tweet_count = 0
			county_tweets = []
			for term in opioid_terms:
				search = api.GetSearch(term, count=100, geocode=geocode)#until='2016-01-01') # Replace happy with your search
				county_tweet_count += len(search)
				for tweet in search:
					county_tweets.append(tweet.AsDict())   #(tweet.location, tweet.text))
			counts.append(county_tweet_count)
			tweets.append(county_tweets)
			df_result = pd.DataFrame(data={'FIPS': codes, 'total_count': counts, 'tweets': tweets})
			df_result.to_csv('tweetsV2_9.csv', sep='\t', encoding='utf-8')
	# print len(fips_codes)
	# print len(counts)
	# print len(tweets)
	df_result = pd.DataFrame(data={'FIPS': fips_codes, 'total_count': counts, 'tweets': tweets})
	df_result.to_csv('tweetsV2_9.csv', sep='\t', encoding='utf-8')
	return df_result
	


# county_db = get_county_db('data/2015_Gaz_counties_national.txt')
# res1 = get_lat_lon_from_FIPS(29051, county_db)
# print res1
# geocode = str(res1[0]) + ',' + str(res1[1]) + ",20mi"
# print geocode
#search = api.GetSearch("OPIOID", count=100, geocode=geocode)#until='2016-01-01') # Replace happy with your search
#print len(search)
# tweets = [i.AsDict() for i in search]


api = twitter.Api(consumer_key=api_key,
  consumer_secret=api_secret,
  access_token_key=access_token,
  access_token_secret=access_token_secret,
  sleep_on_rate_limit=True)

create_twitter_db(api, get_fips_list(), get_related_terms())






