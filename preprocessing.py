import pandas as pd
import numpy as np
import statsmodels.api as sm
import csv
from sklearn import linear_model


import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
from sklearn.linear_model import ElasticNet


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

#CITE: http://scikit-learn.org/stable/modules/linear_model.html

def get_X():
	X_pr = get_X_prescribing_rates()
	X_mort = get_X_mortality()
	X = pd.merge(X_pr, X_mort, on='county_code', how='outer')
	X = X.fillna(value=0)
	return X

def get_X_prescribing_rates():
	X = pd.read_csv('data/pr2008.csv')
	X = X[['FIPS County Code', str(2008)+' Prescribing Rate']].copy() 
	X['2008 Prescribing Rate'] = pd.to_numeric(X['2008 Prescribing Rate'], errors='coerce').fillna(0)
	for i in range(2009, 2017):
		filename = 'data/pr' + str(i) + '.csv'
		X_temp = pd.read_csv(filename) 
		X_temp[str(i)+' Prescribing Rate'] = pd.to_numeric(X_temp[str(i)+' Prescribing Rate'], errors='coerce').fillna(0)#.replace('-', 0.0)
		X = pd.merge(X, X_temp[['FIPS County Code', str(i)+' Prescribing Rate']], on ='FIPS County Code', how='outer')
	X = X.rename(index=str, columns={'FIPS County Code': "county_code"})
	X = X.fillna(value=0)
	X = X.drop_duplicates()

	return X


def get_X_mortality():
	X_full_mortality = pd.read_csv('data/mortality_2008to2015.csv') 
	X_full_mortality.columns  = ['notes', 'county', 'county_code', 'year', 'year_code', 'COD', 'COD_code', 'deaths', 'population', 'crude_rate', 'aa_rate']

	# print X_full_mortality.columns
	# print X_full_mortality['COD_code'].unique()

	X = X_full_mortality.loc[(X_full_mortality['COD_code'] == 'Y14') & (X_full_mortality['year'] == 2008)][['county_code', 'deaths']].copy()
	X.columns = ['county_code','init']

	for year in range(2008, 2016):
		#temp = X_full_mortality[]
		#X['population' + '_' + str(year)] = 
		#TODO: add population data for all years 
		for dc in ['Y14', 'X42', 'X44', 'X41', 'X64', 'Y12', 'X61', 'X62', 'X60', 'X40', 'Y11']:
			#print X_full_mortality['COD_code' == dc]
			X_temp = X_full_mortality.loc[(X_full_mortality['COD_code'] == dc) & (X_full_mortality['year'] == year)][['county_code', 'deaths']].copy()
			X_temp.columns = ['county_code', 'deaths' + '_' + str(year) + '_' + dc]
			X = pd.merge(X, X_temp, on='county_code', how='outer')
	
	X = X.drop(['init'], axis=1)
	X = X.fillna(value=0)
	X_temp = X_full_mortality.loc[(X_full_mortality['year'] == 2015)][['county_code', 'population']].copy()
	X_temp.columns = ['county_code', 'population_2015']
	X = pd.merge(X, X_temp, on='county_code', how='outer')
	X = X.drop_duplicates()
	return X


def get_Y(single_value = True):
	Y_full = pd.read_csv('data/mortality_2016.csv')
	Y_full.columns  = ['notes', 'county', 'county_code', 'year', 'year_code', 'COD', 'COD_code', 'deaths', 'population', 'crude_rate', 'aa_rate']
	
	Y = Y_full[['county_code', 'population']].copy()

	for dc in ['Y14', 'X42', 'X44', 'X41', 'X64', 'Y12', 'X61', 'X62', 'X60', 'X40', 'Y11']:
			#print X_full_mortality['COD_code' == dc]
			Y_temp = Y_full.loc[(Y_full['COD_code'] == dc)][['county_code', 'deaths']].copy()
			Y_temp.columns = ['county_code', dc]
			Y = pd.merge(Y, Y_temp, on='county_code', how='outer')
	
	#TODO (maybe): sum raw death count over all death codes 
	#Y['crude_rate'] = Y['crude_rate'].str.replace(r' \(Unreliable\)', '').astype('double')
	Y = Y.fillna(value=0)
	if single_value:
		Y['total_death_rate'] = Y[['Y14', 'X42', 'X44', 'X41', 'X64', 'Y12', 'X61', 'X62', 'X60', 'X40', 'Y11']].sum(axis=1)
		Y['total_death_rate'] = Y['total_death_rate']/Y['population']
		Y = Y.drop(['population', 'Y14', 'X42', 'X44', 'X41', 'X64', 'Y12', 'X61', 'X62', 'X60', 'X40', 'Y11'], axis=1)
	Y = Y.drop_duplicates()
	return Y 



X = get_X()
Y = get_Y()


# print X.columns
# print X.shape[0]
# print len(X['county_code'].unique())
# print Y.columns
# print Y.shape[0]
# print len(Y['county_code'].unique())

has_x = X['county_code'].values.tolist()
overlap = Y.loc[Y['county_code'].isin(has_x)].copy()
county_list = overlap['county_code'].values.tolist()
X = X.loc[X['county_code'].isin(county_list)].copy()
Y = Y.loc[Y['county_code'].isin(county_list)].copy()
X = X.sort_values(by=['county_code'])
Y = Y.sort_values(by=['county_code'])
X = X.set_index('county_code')
Y = Y.set_index('county_code')
# print X.columns
# X = X.drop(['county_code'], axis=1)
# Y = Y.drop(['county_code'], axis=1)
# print X.columns


X_np = X.as_matrix()
Y_np = Y.as_matrix()
print X_np.shape
print Y_np.shape
print Y_np[0]

write_matrix(X_np, 'X_np')
write_matrix(Y_np, 'Y_np')




#X_train, X_dev, X_test, Y_train, Y_dev, Y_test = get_splits(X, Y)


