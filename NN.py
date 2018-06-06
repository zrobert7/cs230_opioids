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
from sklearn.preprocessing import normalize
from sklearn.pipeline import Pipeline
from matplotlib import pyplot 
from keras import optimizers
from keras.callbacks import TensorBoard
import os
import tensorflow as tf

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
	model.add(Dense(50, input_shape = [98], activation='relu'))
	model.add(Dense(20, input_shape = [98], activation='relu'))
	model.add(Dense(1, activation='relu'))
	model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['mse', 'mae', 'mape'])
	model.summary()
	number_of_iterations = 100
	batch_size = 20
	model.fit(X_train, y_train, batch_size=batch_size, epochs=number_of_iterations, verbose=1, validation_data=(X_test, y_test))


#cite: https://stackoverflow.com/questions/47877475/keras-tensorboard-plot-train-and-validation-scalars-in-a-same-figure
class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='./logs/lr', **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        now = time.strftime("%c")
        training_log_dir = os.path.join(log_dir, str(now) +'_training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, str(now) +'_validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()


#CITE: https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/

def NN1_model(lr=0.001):
	# create model
	model = Sequential()
	model.add(Dense(98, input_dim=98, kernel_initializer='normal', activation='relu'))	
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	adam = optimizers.Adam(lr=lr)#clipvalue=1000, clipnorm=1)
	model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mse', 'mae', 'mape'], )
	return model

def NN1_clip_model(lr=0.001):
	# create model
	model = Sequential()
	model.add(Dense(98, input_dim=98, kernel_initializer='normal', activation='relu'))	
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	adam = optimizers.Adam(lr=lr, clipvalue=1000, clipnorm=1)
	model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mse', 'mae', 'mape'], )
	return model

def NN2_model(lr=0.001):
	# create model
	model = Sequential()
	model.add(Dense(98, input_dim=98, kernel_initializer='normal', activation='relu'))	
	model.add(Dense(20, input_shape = [98], activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	adam = optimizers.Adam()#clipvalue=1000, clipnorm=1)
	model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mse', 'mae', 'mape'], )
	return model

def NN2_clip_model(lr=0.001):
	# create model
	model = Sequential()
	model.add(Dense(98, input_dim=98, kernel_initializer='normal', activation='relu'))	
	model.add(Dense(20, input_shape = [98], activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	adam = optimizers.Adam(clipvalue=1000, clipnorm=1)
	model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mse', 'mae', 'mape'], )
	return model

def NN3_model(lr=0.001):
	# create model
	model = Sequential()
	model.add(Dense(98, input_dim=98, kernel_initializer='normal', activation='relu'))	
	model.add(Dense(50, input_shape = [98], activation='relu'))
	model.add(Dense(20, input_shape = [98], activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	adam = optimizers.Adam()#clipvalue=1000, clipnorm=1)
	model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mse', 'mae', 'mape'], )
	return model

def NN3_clip_model(lr=0.001):
	# create model
	model = Sequential()
	model.add(Dense(98, input_dim=98, kernel_initializer='normal', activation='relu'))	
	model.add(Dense(50, input_shape = [98], activation='relu'))
	model.add(Dense(20, input_shape = [98], activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	adam = optimizers.Adam(clipvalue=1000, clipnorm=1)
	model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mse', 'mae', 'mape'], )
	return model

def try_NN_kfold(X, Y, model_name=NN1_model, lr=0.001):
	seed = 7
	np.random.seed(seed)
	# evaluate model with standardized dataset
	#estimator = KerasRegressor(build_fn=baseline_model, epochs=1, batch_size=5, verbose=1)
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
		
		model = model_name(lr)
		now = time.strftime("%c")
		tbCallBack = keras.callbacks.TensorBoard(log_dir='./logs/'+now, histogram_freq=0, write_graph=True, write_images=True)
		history = model.fit(X[train], Y[train],  validation_split=0.15, epochs=5000, batch_size=len(X), verbose=0, callbacks=[TrainValTensorBoard(write_graph=False)])#[tbCallBack])
		# TODO: maybe use validation_split=0.2,
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


X_np = normalize(read_matrix('X_np'))

Y_np = read_matrix('Y_np')
X_train, X_test, y_train, y_test = data_splitter(X_np, Y_np)
#X_train = normalize(X_train) #is this helpful?
y_train = np.reshape(y_train, (len(y_train), 1))

# #try_NN1(X_train, X_test, y_train, y_test)
# try_NN_kfold(X_np, Y_np, NN1_model)
# try_NN_kfold(X_np, Y_np, NN2_model)
# try_NN_kfold(X_np, Y_np, NN3_model)
# try_NN_kfold(X_np, Y_np, NN1_clip_model)
# try_NN_kfold(X_np, Y_np, NN2_clip_model)
# try_NN_kfold(X_np, Y_np, NN3_clip_model)


lrs = [0.1, 0.01, 0.001, 0.0001]
for lr in lrs:
	try_NN_kfold(X_np, Y_np, NN1_clip_model, lr)
