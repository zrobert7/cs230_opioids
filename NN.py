import pandas as pd
import numpy as np
import statsmodels.api as sm
import time
from sklearn import linear_model
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import ElasticNet
import keras
from keras.layers import Dense
from keras.models import Sequential
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from matplotlib import pyplot 


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

def data_splitter(X, y, test_size=0.1,verbose=True):
	''' 
	CITE: https://github.com/lucko515/fully-connected-nn/blob/master/data_handlers.py
	This fucnction is used to split dataset into training and testing parts
	Input: X- features
			y- labels
			test_size - how much samples from dataset you want to use for testing: Default is 20% - 0.2
			verbose - showing sizes of splited data or not
	Output: X_train, y_train, X_test, y_test - features split into train and test set
	'''

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
	
	if verbose:
		print("X_train size -->", X_train.shape)
		print("y_train size -->", y_train.shape)
		print("X_test size -->", X_test.shape)
		print("y_test size -->", y_test.shape)

	return X_train, X_test, y_train, y_test

def try_NN1(X_train, X_test, y_train, y_test):
	model = Sequential()
	model.add(Dense(50, input_shape = [98], activation='sigmoid'))
	model.add(Dense(20, input_shape = [98], activation='sigmoid'))
	model.add(Dense(1, activation='relu'))
	model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])
	model.summary()
	number_of_iterations = 100
	batch_size = 20
	model.fit(X_train, y_train, batch_size=batch_size, epochs=number_of_iterations, verbose=2, validation_data=(X_test, y_test))


#CITE: https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(98, input_dim=98, kernel_initializer='normal', activation='relu'))	
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae', 'mape'])
	return model

def try_NN2(X, Y):
	seed = 7
	np.random.seed(seed)
	# evaluate model with standardized dataset
	estimator = KerasRegressor(build_fn=baseline_model, epochs=1, batch_size=5, verbose=1)
	kfold = KFold(n_splits=10, random_state=seed, shuffle=True)

	mse_scores = []
	mae_scores = []
	for train, test in kfold.split(X, Y):
	  # create model
		# model = Sequential()
		# model.add(Dense(12, input_dim=8, activation='relu'))
		# model.add(Dense(8, activation='relu'))
		# model.add(Dense(1, activation='sigmoid'))
		# # Compile model
		# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		# # Fit the model
		# model.fit(X[train], Y[train], epochs=150, batch_size=10, verbose=0)
		


		model = baseline_model()
		now = time.strftime("%c")
		tbCallBack = keras.callbacks.TensorBoard(log_dir='./logs/'+now, histogram_freq=0, write_graph=True, write_images=True)
		history = model.fit(X[train], Y[train], epochs=3000, batch_size=len(X), verbose=0, callbacks=[tbCallBack])

		# evaluate the model
		scores = model.evaluate(X[test], Y[test], verbose=0)
		print("%s: %.2f" % (model.metrics_names[1], scores[1]))
		print("%s: %.2f" % (model.metrics_names[2], scores[2]))
		mse_scores.append(scores[1])
		mae_scores.append(scores[2])
	print("%.2f (+/- %.2f)" % (np.mean(mse_scores), np.std(mse_scores)))
	print("%.2f (+/- %.2f)" % (np.mean(mae_scores), np.std(mae_scores)))

	# results = cross_val_score(estimator, X, Y, cv=kfold)
	# print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
	# model = baseline_model()
	# now = time.strftime("%c")
	# tbCallBack = keras.callbacks.TensorBoard(log_dir='./logs/'+now, histogram_freq=0, write_graph=True, write_images=True)
	# history = model.fit(X, Y, validation_split=0.2, epochs=2500, batch_size=len(X), verbose=1, callbacks=[tbCallBack])


X_np = read_matrix('X_np')
Y_np = read_matrix('Y_np')
X_train, X_test, y_train, y_test = data_splitter(X_np, Y_np)
y_train = np.reshape(y_train, (len(y_train), 1))

try_NN2(X_np, Y_np)
