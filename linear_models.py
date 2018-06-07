
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn import linear_model
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import normalize


def try_LR(X, y):

	# Note the difference in argument order
	model = sm.OLS(y, X).fit()
	predictions = model.predict(X) # make the predictions by the model

	# Print out the statistics
	model.summary()

def try_lasso(X, y):
	alpha = 0.1
	n_samples = X.shape[0]
	print(str(n_samples) + " samples")
	X_train, y_train = X[:n_samples // 2], y[:n_samples // 2]
	X_test, y_test = X[n_samples // 2:], y[n_samples // 2:]
	lasso = linear_model.Lasso(alpha = 0.1, max_iter=10000)
	y_pred_lasso = lasso.fit(X_train, y_train).predict(X_test)
	r2_score_lasso = r2_score(y_test, y_pred_lasso)
	print(lasso)
	print("r^2 on test data : %f" % r2_score_lasso)

def try_elastic(X, y):
	alpha = 0.1
	n_samples = X.shape[0]
	print(str(n_samples) + " samples")
	X_train, y_train = X[:n_samples // 2], y[:n_samples // 2]
	X_test, y_test = X[n_samples // 2:], y[n_samples // 2:]
	enet = ElasticNet(alpha=alpha, l1_ratio=0.7)

	y_pred_enet = enet.fit(X_train, y_train).predict(X_test)
	r2_score_enet = r2_score(y_test, y_pred_enet)
	print(enet)
	print("r^2 on test data : %f" % r2_score_enet)

def try_ridge(X,y):
	alpha = 0.1
	n_samples = X.shape[0]
	print(str(n_samples) + " samples")
	X_train, y_train = X[:n_samples // 2], y[:n_samples // 2]
	X_test, y_test = X[n_samples // 2:], y[n_samples // 2:]
	ridge = linear_model.Ridge (alpha = .5)

	y_pred_ridge = ridge.fit(X_train, y_train).predict(X_test)
	r2_score_ridge = r2_score(y_test, y_pred_ridge)
	print(ridge)
	print("r^2 on test data : %f" % r2_score_ridge)

def try_SGD(X, y):

	n_samples = X.shape[0]
	print(str(n_samples) + " samples")
	X_train, y_train = X[:n_samples // 2], y[:n_samples // 2]
	X_test, y_test = X[n_samples // 2:], y[n_samples // 2:]
	ridge = linear_model.SGDRegressor (alpha = .5)

	y_pred_ridge = ridge.fit(X_train, y_train).predict(X_test)
	r2_score_ridge = r2_score(y_test, y_pred_ridge)
	print(ridge)
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

X_np = normalize(read_matrix('X_np'))
Y_np = read_matrix('Y_np')
X_np = pd.read_csv('X_embedded_np.csv', sep=' ',header=None).as_matrix()
X_np = normalize(X_np, axis=0)
X_np = pd.read_csv('X_tweet_counts.csv', sep=' ',header=None).as_matrix().T
print X_np.shape

try_lasso(X_np, Y_np)
try_elastic(X_np, Y_np)
try_ridge(X_np, Y_np)
