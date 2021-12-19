# loading the models for local and global embedding
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses
import generate_input_RQ as gt
import myModel_RQ_selectivity as m
from sklearn.model_selection import train_test_split
import plot as p

def load_and_run_model(file_x, file_y):
	x = np.load(file_x)
	y = np.load(file_y)
	#for i in range(y.shape[0]):
	#	if (y[i] > 0.0):
	#		y[i] += 0.2
	maximum = np.amax(y, axis=(0))
	y_nor = gt.gh.nor_y_ab(y,100,0.0,maximum)
	freq = gt.gh.count_frequency_1(y_nor)
	p.plot_freq(freq)

	model_rq = m.RQ_sel_2Input_CNN2L_CNN2L_noBN_DENSE2L(32,32,3072,1024,3072,1024)
	#model_rq = m.RQ_sel_2Input_DENSE3L_DENSE2L(32,32,2048,1024,2048,1024,256)
	model_rq.compile(optimizer='adam', loss=losses.MeanSquaredError())
	#model_rq.compile(optimizer='adam', loss=losses.MeanAbsolutePercentageError())
	print("Splitting training and test set...")
	X_train_full, X_test, y_train_full, y_test = train_test_split(x, y_nor, test_size=0.2)
	X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size=0.2)
	X_loc_train = X_train[:,:,:,0:3]
	X_loc_valid = X_valid[:,:,:,0:3]
	X_loc_test = X_test[:,:,:,0:3]
	X_glo_train = X_train[:,:,:,3:6]
	X_glo_valid = X_valid[:,:,:,3:6]
	X_glo_test = X_test[:,:,:,3:6]
	callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
	
	history = model_rq.fit([X_loc_train, X_glo_train], y_train, epochs=40, batch_size=8, shuffle=True, #validation_data=([X_loc_valid, X_glo_valid], y_valid))
callbacks=[callback], validation_data=([X_loc_valid, X_glo_valid], y_valid))
	model_rq.summary()
	y_pred = model_rq.predict([X_loc_test, X_glo_test])
	diff = abs(y_test - y_pred)
	print ("MAE norm: ", np.mean(diff))
	mape0, freq0, num0, mae_zero0, freq_zero0, num_zero0 = mape_error_zero (y_test, y_pred)
	print ("MAPE norm non zero: ", mape0, freq0, num0)
	print ("MAE norm zero: ", mae_zero0, freq_zero0, num_zero0)

	# Denormalized prediction
	y_test_den = gt.gh.denorm_y_ab(y_test,100,0.0,maximum)
	#for i in range(y_test_den.shape[0]):
	#	if (y_test_den[i] > 0.0):
	#		y_test_den[i] -= 0.2
	#	if (y_test_den[i] < 0.0):
	#		y_test_den[i] = 0.0
	#y_test_den = gt.gh.denorm_y_ab(y_test_den,200,0.0,0.5)
	#
	y_pred_den = gt.gh.denorm_y_ab(y_pred,100,0.0,maximum)
	#for i in range(y_pred_den.shape[0]):
	#	if (y_pred_den[i] > 0.0):
	#		y_pred_den[i] -= 0.2
	#	if (y_pred_den[i] < 0.0):
	#		y_pred_den[i] = 0.0
	#y_pred_den = gt.gh.denorm_y_ab(y_pred,200,0.0,0.5)
	diff_den = abs(y_test_den - y_pred_den)
	print ("MAE denor: ", np.mean(diff_den))
	mape, freq, num, mae_zero, freq_zero, num_zero = mape_error_zero (y_test_den, y_pred_den)
	print ("MAPE denor non zero: ", mape, freq, num)
	print ("MAE denor zero: ", mae_zero, freq_zero, num_zero)

	return model_rq, y_test_den, y_pred_den, history, X_loc_test, X_glo_test

def run_model():
	local_enc = keras.models.load_model('model/model_3072_CNNDense_newDatasets_SMLG')
	global_enc = keras.models.load_model('model/model_2048_CNNDense_newDatasets_SMLG_global_new')

	x, y = gt.gen_rq_input_from_file(local_enc, global_enc, 'rq/rq_newDatasets.csv', 'mbr/mbr_alldatasets.csv', 'rq/rq_result.csv', 'dataset-summaries.csv', 'histograms/new_datasets/', ';')

	#
	print("Inizilizing model DENSE local...")
	filter = 128
	model_rq = m.RQ_sel_DENSE(32,32,7,filter)
	model_rq.compile(optimizer='adam', loss=losses.MeanSquaredError())
	print("Splitting training and test set...")
	X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

	# training
	print("Training model...")
	history = model_rq.fit(X_train, y_train, batch_size=8, epochs=0, shuffle=True, validation_data=(X_test, y_test))
	return x, y, X_train, X_test, y_train, y_test, model_rq, history

def mape_error_zero (y, predict):
	delta_zero = 0.0
	zero = 0
	non_zero = 0
	delta = 0.0
	p = 0
	freq_zero = np.zeros((7))
	freq = np.zeros((7))
	for i in range(y.shape[0]):
		if (y[i] == 0.0):
			zero += 1
			delta_zero += predict[i]
			if (predict[i] == 0.0):
				freq_zero[6] += 1
			if (predict[i] < 0.000001):
				freq_zero[5] += 1
			elif (predict[i] < 0.00001):
				freq_zero[4] += 1
			elif (predict[i] < 0.0001):
				freq_zero[3] += 1
			elif (predict[i] < 0.001):
				freq_zero[2] += 1
			elif (predict[i] < 0.01):
				freq_zero[1] += 1
			elif (predict[i] < 0.1):
				freq_zero[0] += 1
			if (zero < 10 and predict[i] > 0.1):
				print(y[i], predict[i])
			#if (predict[i] < 0.0):
			#	predict[i] = 0.0
		else:
			non_zero += 1
			delta += abs(y[i] - predict[i])/y[i]
			if (abs(y[i] - predict[i])/y[i] < 0.00001):
				freq[6] += 1
			elif (abs(y[i] - predict[i])/y[i] < 0.0001):
				freq[5] += 1
			elif (abs(y[i] - predict[i])/y[i] < 0.001):
				freq[4] += 1
			elif (abs(y[i] - predict[i])/y[i] < 0.01):
				freq[3] += 1
			elif (abs(y[i] - predict[i])/y[i] < 0.1):
				freq[2] += 1
			elif (abs(y[i] - predict[i])/y[i] < 1):
				freq[1] += 1
			elif (abs(y[i] - predict[i])/y[i] < 10):
				freq[0] += 1
				
			if (p < 10 and abs(y[i] - predict[i])/y[i] > 50):
				print(y[i], predict[i])
				p += 1
	return delta/non_zero, freq, non_zero, delta_zero/zero, freq_zero, zero
def zero_one (y):
	y_zero_one = np.zeros((y.shape[0]))
	for i in range(y.shape[0]):
		if (y[i] > 0.0):
			y_zero_one[i] = 1
	return y_zero_one
