#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import generate_histogram as gh
import myModel_RQ_selectivity as mm
#
# Model definition
#
# Model creating...
def create_model_RQ_sel(dimx,dimy,dimz):
	print("Inizilizing Autoencoder DENSE local...")
	m_rq_sel = RQ_sel_DENSE(dimx,dimy,dimz,1024)
	m_rq_sel.compile(optimizer='adam', loss=losses.MeanSquaredError())
	return m_rq_sel

# Model training...
def train_model_RQ_sel(mod, X, y)
	print('Data points: ',X.shape)
	# splitting train and test 0.2
	print("Splitting training and test set...")
	X_train, X_test, y_train_sel, y_test_sel = train_test_split(X, y, test_size=0.2)
	# training
	print("Training model RQ selectivity...")
	history = mod.fit(X_train, y_train_sel, batch_size=8, epochs=200, validation_data=(X_test, y_test_sel))
	return history, mod, X_train, X_test, y_train_sel, y_test_sel 

