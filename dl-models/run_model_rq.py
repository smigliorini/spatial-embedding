# loading the models for local and global embedding
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses
import generate_input_RQ as gt
import myModel_RQ_selectivity as m
from sklearn.model_selection import train_test_split

def load_and_run_model(file_x, file_y):
	x = np.load(file_x)
	y = np.load(file_y)
	y_nor = rq.gt.gh.nor_y_ab(y) 

	model_rq = m.RQ_sel_2Input_DENSE2L_DENSE1L(32,32,3,2,16,32,64)
	model_rq.compile(optimizer='adam', loss=losses.MeanSquaredError())
	print("Splitting training and test set...")
	X_train_full, X_test, y_train_full, y_test = train_test_split(x, y, test_size=0.2)
	X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)
	X_loc_train = X_train[:,:,:,0:3]
	X_loc_valid = X_valid[:,:,:,0:3]
	X_loc_test = X_test[:,:,:,0:3]
	X_glo_train = X_train[:,:,:,3:7]
	X_glo_valid = X_valid[:,:,:,3:7]
	X_glo_test = X_test[:,:,:,3:7]
	history = model_rq.fit([X_loc_train, X_glo_train], [y_train, y_train], epochs=20, batch_size=8, shuffle=True, validation_data=([X_loc_valid, X_glo_valid], [y_valid, y_valid])
	
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
	for i in range(y.shape[0]):
		if (y[i] == 0.0):
			zero += 1
			delta_zero += predict[i]
			if (zero < 50):
				print(y[i], predict[i])
		else:
			non_zero += 1
			delta += abs(y[i] - predict[i])/y[i]
			if (p < 50 and abs(y[i] - predict[i])/y[i] > 50):
				print(y[i], predict[i])
				p += 1
	return delta/non_zero, non_zero, delta_zero/zero, zero
def zero_one (y):
	y_zero_one = np.zeros((y.shape[0]))
	for i in range(y.shape[0]):
		if (y[i] > 0.0):
			y_zero_one[i] = 1
	return y_zero_one
