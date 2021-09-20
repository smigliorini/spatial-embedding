#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import generate_histogram as gh
#
# generation of random training set
# a: local histograms
# g: global histograms
# b: rq selectivity
# r: range query histogram
def create_h(n,dimx,dimy,dimz,norm_type,distr):
	#n = 2500
	#dimx = 128
	#dimy = 128
	#dimz = 6
	print("Generating ", n, " random histograms (",dimx,"x",dimy,"x",dimz,") ...")
	if (norm_type == 0):
		print("Method for reducing the range of the feature values: standardization")
	else:
		print("Method for reducing the range the feature values: normalization min-max")
	if (distr == 0):
		print("Distribution: UNIFORM")
	else:
		if (distr == 1):
			print("Distribution: DIAGONAL")
		else:
			print("Distribution: MIX")
	a, g, b, r  = gh.generate(n,dimx,dimy,dimz,distr) # 0 for uniform, 1 for diagonal
	# standardization of data z = (x - avg)/mse results are in the range -1, 1
	# or normalization of data z = (x - min)/(max - min) 
	# outliers can be outside the range
	if (norm_type == 0):
		print("Standardizing data...")
		a_scaled, g_scaled, b_scaled = gh.std(a,g,b)
	else:
		print("Normalizing data...")
		a_scaled, g_scaled, b_scaled = gh.nor(a,g,b)
	return a_scaled, g_scaled, b_scaled, r

# model definition

# Autoencoder
def auto_encoder(loc,glo,dimx,dimy,dimz,ldim_loc,ldim_glo,a,g,b,r):
	import myAutoencoder as enc
	#dimx = 128
	#dimy = 128
	#dimz = 6
	if (loc == 1):
		latent_dim = ldim_loc
		print("Inizilizing Autoencoder local...")
		print("Embedding dim: ", latent_dim)
		ae_loc = enc.Autoencoder_local(latent_dim,dimx,dimy,dimz)
		ae_loc.compile(optimizer='adam', loss=losses.MeanSquaredError())
		#ae_loc.compile(optimizer='adam', loss='binary_crossentropy')

		# splitting train and test 0.2
		print("Splitting training and test set...")
		# local histograms
		X_train_hist, X_test_hist, X_train_range, X_test_range, y_train_card, y_test_card = train_test_split(a, r, b, test_size=0.2)
		# training
		print("Training Autoencoder local...")
		ae_loc.fit(X_train_hist, X_train_hist, epochs=30, shuffle=True, validation_data=(X_test_hist, X_test_hist))
	else:
		print("Skip local autoencoder")
	if (glo == 1):
		latent_dim = ldim_glo
		print("Inizilizing Autoencoder global...")
		print("Embedding dim: ", latent_dim)
		ae_glo = enc.Autoencoder_global(latent_dim,dimx,dimy)
		ae_glo.compile(optimizer='adam', loss=losses.MeanSquaredError())
		#ae_glo.compile(optimizer='adam', loss='binary_crossentropy')
		# splitting train and test 0.2
		print("Splitting training and test set...")
		# global histograms
		X_train_global, X_test_global = train_test_split(g, test_size=0.2)
		# training
		print("Training Autoencoder global...")
		ae_glo.fit(X_train_global, X_train_global, epochs=30, shuffle=True, validation_data=(X_test_global, X_test_global))
	else:
		print("Skip global autoencoder")
	if (loc == 1):
		if (glo == 1):
			return ae_loc, X_train_hist, X_test_hist, ae_glo, X_train_global, X_test_global
		else:
			return ae_loc, X_train_hist, X_test_hist
	if (glo == 1):
		return ae_glo, X_train_global, X_test_global
