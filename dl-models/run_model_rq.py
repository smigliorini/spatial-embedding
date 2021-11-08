# loading the models for local and global embedding
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses
import gen_inputRQ_test as gt
import myModel_RQ_selectivity as m
from sklearn.model_selection import train_test_split

def run_model():
	local_enc = keras.models.load_model('model/model_3072_CNNDense_newDatasets_SMLG')
	global_enc = keras.models.load_model('model/model_2048_CNNDense_newDatasets_SMLG_global')

	x, y = gt.gen_rq_input_from_file(local_enc, global_enc, 'rq/rq_newDatasets.csv', 'mbr/mbr_alldatasets.csv', 'rq/rq_result.csv', 'histograms/new_datasets', ',', ';')

	#
	print("Inizilizing model DENSE local...")
	filter = 128
	model_rq = m.RQ_sel_DENSE(32,32,7,filter)
	model_rq.compile(optimizer='adam', loss=losses.MeanSquaredError())
	print("Splitting training and test set...")
	X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

	# training
	print("Training model...")
	history = model_rq.fit(X_train, y_train, batch_size=16, epochs=30, shuffle=True, validation_data=(X_test, y_test))
	return x, y, X_train, X_test, y_train, y_test, model_rq, history
