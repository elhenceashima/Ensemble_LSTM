from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from math import sqrt
from matplotlib import pyplot
import numpy
from numpy import concatenate
import statistics
from scipy import stats
from scipy.stats import norm
from collections import deque


TRAIN_SIZE = 200
TEST_SIZE = 200
updates = 1
ALPHA = 0.2
WINDOW_SIZE = 7
CUTOFF = 4
N = 3

# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
	df = DataFrame(data)
	columns = [df.shift(i) for i in range(1, lag+1)]
	columns.append(df)
	df = concat(columns, axis=1)
	df.fillna(0, inplace=True)
	return df

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

# scale train and test data to [-1, 1]
def scale(train, test):
	# fit scaler
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaler = scaler.fit(train)
	# transform train
	train = train.reshape(train.shape[0], train.shape[1])
	train_scaled = scaler.transform(train)
	# transform test
	test = test.reshape(test.shape[0], test.shape[1])
	test_scaled = scaler.transform(test)
	return scaler, train_scaled, test_scaled

# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
	new_row = [x for x in X] + [value]
	array = numpy.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]

def fit_lstm(train, batch_size, nb_epoch, neurons):
	X, y = train[:, 0:-1], train[:, -1]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	model = Sequential()
	model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True, return_sequences=True))
	model.add(LSTM(neurons, stateful=True, return_sequences=True, dropout = 0.3))
	model.add(LSTM(neurons, stateful=True, return_sequences=False, dropout = 0.3))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	
	for i in range(nb_epoch):
		model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
		model.reset_states()
	
	return model

def update_model(model, train, batch_size, updates):
	X, y = train[:, 0:-1], train[:, -1]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	for i in range(updates):
		model.fit(X, y, nb_epoch=1, batch_size=batch_size, verbose=0, shuffle=False)
		model.reset_states()

# make a one-step forecast
def forecast_lstm(model, batch_size, X):
	X = X.reshape(1, 1, len(X))
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]

def normalizing(imse, weight):
	cursum = 0
	for i in range(N):
		cursum += imse[i]
	for i in range(N):
		weight[i] = imse[i]/cursum

# load dataset
series = read_csv('real_5.csv', header=0, usecols=['value'])
#print(series)

# transform data to be stationary
raw_values = series.values
diff_values = difference(raw_values, 1)

# transform data to be supervised learning
supervised = timeseries_to_supervised(diff_values, 1)
supervised_values = supervised.values

# split data into train and test-sets
train, test = supervised_values[0:TRAIN_SIZE], supervised_values[TRAIN_SIZE:]
#train, test = supervised_values[0:-TEST_SIZE], supervised_values[-TEST_SIZE:]

# transform the scale of the data
scaler, train_scaled, test_scaled = scale(train, test)

# fit the model
lstm_model = list()
for i in range(N):
	lstm_model.append(fit_lstm(train_scaled, 1, 5, 16))																									

# forecast the entire training dataset to build up state for forecasting
train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)

for i in range(N):
	lstm_model[i].predict(train_reshaped, batch_size=1)

m = 0;
n = 0;
S = 0;

for x in train[:,-1]:
    prev_mean = m
    n = n + 1
    m = m + (x-m)/n
    S = S + (x-m)*(x-prev_mean)

m = m[0]
S = S[0]
n = n

# walk-forward validation on the test data
#predictions = list()	
predictions = [[] for x in range(0,N)]
#sq_error = [[] for x in range(0,N)]
mse = [N] * N
imse = [1/N] * N
weight = [0] * N
given_pred = list()

#my_q = deque( ["Normal"] * WINDOW_SIZE)
train_copy = numpy.copy(train_scaled)

for i in range(len(test_scaled)):
	# make one-step forecast
	X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
	yhat = list()
	dummy = 0

	for x in range(N):
		yhat.append(forecast_lstm(lstm_model[x], 1, X))

	# invert scaling
	for x in range(N):
		yhat[x] = invert_scale(scaler, X, yhat[x])

	# invert differencing
	for x in range(N):
		yhat[x] = inverse_difference(raw_values, yhat[x], len(test_scaled)+1-i)

	expected = raw_values[len(train) + i + 1]

	# store forecast
	for x in range(N):
		predictions[x].append(yhat[x])

	normalizing(imse, weight)

	for x in range(N):
		dummy = dummy + (weight[x]*yhat[x])
	given_pred.append(dummy)

	for x in range(N):
		#sq_error[x].append(abs(yhat[x]-expected)*abs(yhat[x]-expected))
		mse[x] = mse[x] + (abs(yhat[x]-expected)*abs(yhat[x]-expected))
		imse[x] = 1/mse[x]

	# add to training set
	train_copy = concatenate((train_copy, test_scaled[i,:].reshape(1, -1)))
	#update_model(lstm_model, train_copy, 1, updates)
	for x in range(N):
		update_model(lstm_model[x], train_copy, 1, updates)

	print('Month=%d, Expected=%f' % (i+1, expected))
	print('Predicted=%s' % given_pred[-1])
	'''et = expected - yhat
	pval = stats.norm.cdf((et - m)/sqrt(S/n) , loc = 0, scale = 1)
	print(pval)
	#print(m)
	#print(S)
	
	prev_mean = m
	n = n + 1
	m = m + ((expected - m)/n)
	S = S + ((expected - m)*(expected - prev_mean))

	if(pval < ALPHA or pval > (1 - ALPHA)):
		print("Suspicious")
		if(my_q.count("Suspicious") >= CUTOFF):
			print("Suspicious Change Point")
		else:
			print("Suspicious Outlier")
		my_q.append("Suspicious")
		my_q.popleft()

	else:
		print("Normal")
		update_model(lstm_model, train_copy, 1, updates)
		my_q.append("Normal")
		my_q.popleft()
	'''


# report performance
rmse = sqrt(mean_squared_error(raw_values[TRAIN_SIZE:-1], given_pred))
print('Test RMSE: %.3f' % rmse)
# line plot of observed vs predicted
pyplot.figure(1)
pyplot.plot(raw_values[TRAIN_SIZE:])
pyplot.plot(given_pred)
pyplot.figure(2)
pyplot.plot(raw_values[TRAIN_SIZE:])
for i in range(N):
	pyplot.plot(predictions[i])
pyplot.show()