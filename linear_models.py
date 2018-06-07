
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn import linear_model
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import normalize
from sklearn import linear_model 
from sklearn import metrics
from sklearn.metrics import r2_score

def try_LR(X, y):

	# Note the difference in argument order
	model = sm.OLS(y, X).fit()
	predictions = model.predict(X) # make the predictions by the model

	# Print out the statistics
	print model.summary()

def try_linear_regression(X, y):
	n_samples = X.shape[0]
	test_threshold = 547 #68
	X_train, y_train = X[:test_threshold], y[:test_threshold]
	X_test, y_test = X[test_threshold:], y[test_threshold:]

	lm = linear_model.LinearRegression()
	model = lm.fit(X_train,y_train)
	predictions = lm.predict(X_train)
	print "MSE train: " + str(metrics.mean_squared_error(y_train, predictions))
	print "R2 train: " + str(metrics.r2_score(y_train, predictions))
	predictions = lm.predict(X_test)
	print "MSE test: " + str(metrics.mean_squared_error(y_test, predictions))
	print "R2 test: " + str(metrics.r2_score(y_test, predictions))
	print X_train.shape


def try_lasso(X, y):
	alpha = 0.1
	n_samples = X.shape[0]
	print(str(n_samples) + " samples")
	test_threshold = 547 #68
	X_train, y_train = X[:test_threshold], y[:test_threshold]
	X_test, y_test = X[test_threshold:], y[test_threshold:]
	lasso = linear_model.Lasso(alpha = 0.1, max_iter=10000)
	model = lasso.fit(X_train, y_train)
	predictions = model.predict(X_train)
	print "MSE train: " + str(metrics.mean_squared_error(y_train, predictions))
	print "R2 train: " + str(metrics.r2_score(y_train, predictions))
	predictions = model.predict(X_test)
	print "MSE test: " + str(metrics.mean_squared_error(y_test, predictions))
	print "R2 test: " + str(metrics.r2_score(y_test, predictions))
	r2_score_lasso = r2_score(y_test, predictions)
	print("r^2 on test data : %f" % r2_score_lasso)

def try_elastic(X, y):
	alpha = 0.1
	n_samples = X.shape[0]
	print(str(n_samples) + " samples")
	test_threshold = 547 #68
	X_train, y_train = X[:test_threshold], y[:test_threshold]
	X_test, y_test = X[test_threshold:], y[test_threshold:]
	enet = ElasticNet(alpha=alpha, l1_ratio=0.7)


	model= enet.fit(X_train, y_train)
	predictions = model.predict(X_train)
	print "MSE train: " + str(metrics.mean_squared_error(y_train, predictions))
	print "R2 train: " + str(metrics.r2_score(y_train, predictions))
	predictions = model.predict(X_test)
	print "MSE test: " + str(metrics.mean_squared_error(y_test, predictions))
	print "R2 test: " + str(metrics.r2_score(y_test, predictions))
	r2_score_enet = r2_score(y_test, predictions)
	print("r^2 on test data : %f" % r2_score_enet)

def try_ridge(X,y):
	alpha = 0.1
	n_samples = X.shape[0]
	print(str(n_samples) + " samples")
	test_threshold = 547 #68
	X_train, y_train = X[:test_threshold], y[:test_threshold]
	X_test, y_test = X[test_threshold:], y[test_threshold:]
	ridge = linear_model.Ridge (alpha = .5)

	model = ridge.fit(X_train, y_train)
	predictions = model.predict(X_train)
	print "MSE train: " + str(metrics.mean_squared_error(y_train, predictions))
	print "R2 train: " + str(metrics.r2_score(y_train, predictions))
	predictions = model.predict(X_test)
	print "MSE test: " + str(metrics.mean_squared_error(y_test, predictions))
	print "R2 test: " + str(metrics.r2_score(y_test, predictions))
	r2_score_ridge = r2_score(y_test, predictions)
	print("r^2 on test data : %f" % r2_score_ridge)

def try_SGD(X, y):

	n_samples = X.shape[0]
	print(str(n_samples) + " samples")
	test_threshold = 547 #68
	X_train, y_train = X[:test_threshold], y[:test_threshold]
	X_test, y_test = X[test_threshold:], y[test_threshold:]
	ridge = linear_model.SGDRegressor (alpha = .5)

	model = ridge.fit(X_train, y_train)
	predictions = model.predict(X_train)
	print "MSE train: " + str(metrics.mean_squared_error(y_train, predictions))
	print "R2 train: " + str(metrics.r2_score(y_train, predictions))
	predictions = model.predict(X_test)
	print "MSE test: " + str(metrics.mean_squared_error(y_test, predictions))
	print "R2 test: " + str(metrics.r2_score(y_test, predictions))
	r2_score_ridge = r2_score(y_test, predictions)
	print("r^2 on test data : %f" % r2_score_ridge)

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

#X_np = normalize(read_matrix('X_np'))
Y_np = read_matrix('Y_np')
X_np = pd.read_csv('X_embedded_np.csv', sep=' ',header=None).as_matrix()
X_np = normalize(X_np, axis=0)
#X_np = pd.read_csv('X_tweet_counts.csv', sep=' ',header=None).as_matrix().T
print X_np.shape

#try_LR(X_np, Y_np)
print "LinearRegression"
try_linear_regression(X_np, Y_np)
print "Lasso"
try_lasso(X_np, Y_np)
print "ElasticNet"
try_elastic(X_np, Y_np)
print "Ridge"
try_ridge(X_np, Y_np)
#try_SGD(X_np, Y_np)
