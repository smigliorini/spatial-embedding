# loading the models for local and global embedding
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses
import generate_input_RQ as gt
import myModel_RQ_selectivity as m
from sklearn.model_selection import train_test_split

def mape_error_zero (y, predict,c):
	predict = np.expm1(predict)/c
	y = np.expm1(y)/c
	zeros = predict[y == 0.0]
	y_not_zero_idx = y != 0.0
	deltas = abs(y[y_not_zero_idx] - predict[y_not_zero_idx])/y[y_not_zero_idx]
	return np.mean(deltas), deltas.shape[0], np.mean(zeros), zeros.shape[0]


def zero_one (y):
	return (y > 0.0)*1

def gen_model_single_dense(n_nodes,dropout = 0.5):
	return tf.keras.Sequential([
		# INPUT layer
		layers.Input(shape=(32, 32, 7)),
		layers.Flatten(),
		layers.Dense(n_nodes, activation='relu'),
		layers.BatchNormalization(),
		layers.Dropout(dropout),
		layers.Dense(1, activation="linear")
	])

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


def run_models(x: np.ndarray, y: np.ndarray):
	"""
	count = 100000
	:param x: np.ndarray(count,32,32,7) (ex. x_100000_rq.npy);
	:param y: np.ndarray(count) (ex. y_100000_rq.npy);
	:return:
	"""
	#local_enc = keras.models.load_model('model/model_3072_CNNDense_newDatasets_SMLG')
	#global_enc = keras.models.load_model('model/model_2048_CNNDense_newDatasets_SMLG_global_new')

	#x, y = gt.gen_rq_input_from_file(local_enc, global_enc, 'rq/rq_newDatasets.csv', 'mbr/mbr_alldatasets.csv', 'rq/rq_result.csv', 'dataset-summaries.csv', 'histograms/new_datasets/', ';')

	print("Splitting training and test set...")
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

	for c_exp in range (0,5):
		c = pow(10,c_exp)
		for units_exp in range (6,10):
			print(F"Scaling y with log( 1 + {c} * y / max(y) ) / log( 2 )")
			y_train_log = np.log1p( c*y_train/np.max(y_train) )/np.log1p(1)
			y_test_log = np.log1p( c*y_test/np.max(y_test) )/np.log1p(1)
			units = pow(2,units_exp)
			print(F"Initializing model DENSE with {units} nodes")
			model_rq = gen_model_single_dense(units)
			model_rq.compile(optimizer='adam', loss=losses.MeanSquaredError())
			print("Training model...")
			model_rq.fit(x_train, y_train_log, batch_size=256, epochs=50, shuffle=True, validation_data=(x_test, y_test_log))
			predict = model_rq.predict(x_test)
			print(F"Errors: {mape_error_zero(y_test_log, predict, c)}")
