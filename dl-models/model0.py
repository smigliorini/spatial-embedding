#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import generate_histogram as gh
import myAutoencoder as enc
#
# generation of random training set
# a: local histograms
# g: global histograms
# b: rq selectivity
# r: range query histogram
def create_h(n,dimx,dimy,dimz,norm_type,distr):
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
		a_scaled = gh.std_a(a)
		g_scaled = gh.std_g(g)
		b_scaled = gh.std_b(b)
	else:
		print("Normalizing data...")
		a_scaled = gh.nor_a(a)
		g_scaled = gh.nor_g(g)
		b_scaled = gh.nor_b(b)
	return a_scaled, g_scaled, b_scaled, r

# model definition

# Autoencoder
def auto_encoder(type,loc,glo,dimx,dimy,dimz,ldim_loc,ldim_glo,a,g,b,r):
	# latent_dim is equal to the size of the embedding representation of the histogram for DENSE network
	# while for CNN it is equal to the number of values in each cell of the embedding, thus the size of
	# the embedding is obtained by the formula: dimx/4 * dimy/4 * latent_dim
	if (loc == 1):
		latent_dim = ldim_loc
		if (type == 0):
			print("Embedding dim: ", latent_dim)
			print("Inizilizing Autoencoder DENSE local...")
			ae_loc = enc.Autoencoder_local(latent_dim,dimx,dimy,dimz)
		else:
			print("Embedding dim: ", latent_dim*dimx/4*dimy/4)
			print("Inizilizing Autoencoder CNN local...")
			ae_loc = enc.AutoencoderCNN_local(latent_dim,dimx,dimy,dimz)
		ae_loc.compile(optimizer='adam', loss=losses.MeanSquaredError())
		#ae_loc.compile(optimizer='adam', loss='binary_crossentropy')

		# splitting train and test 0.2
		print("Splitting training and test set...")
		# local histograms
		X_train_hist, X_test_hist, X_train_range, X_test_range, y_train_card, y_test_card = train_test_split(a, r, b, test_size=0.2)
		# training
		print("Training Autoencoder local...")
		ae_loc.fit(X_train_hist, X_train_hist, batch_size=16, epochs=20, shuffle=True, validation_data=(X_test_hist, X_test_hist))
	else:
		print("Skip local autoencoder")
	if (glo == 1):
		latent_dim = ldim_glo
		if (type == 0):
			print("Embedding dim: ", latent_dim)
			print("Inizilizing Autoencoder DENSE global...")
			ae_glo = enc.Autoencoder_global(latent_dim,dimx,dimy)
		else:
			print("Embedding dim: ", latent_dim*dimx/4*dimy/4)
			print("Inizilizing Autoencoder CNN global...")
			ae_glo = enc.AutoencoderCNN_global(latent_dim,dimx,dimy,1)
		ae_glo.compile(optimizer='adam', loss=losses.MeanSquaredError())
		#ae_glo.compile(optimizer='adam', loss='binary_crossentropy')
		# splitting train and test 0.2
		print("Splitting training and test set...")
		# global histograms
		X_train_global, X_test_global = train_test_split(g, test_size=0.2)
		# training
		print("Training Autoencoder global...")
		ae_glo.fit(X_train_global, X_train_global, batch_size=32, epochs=12, shuffle=True, validation_data=(X_test_global, X_test_global))
	else:
		print("Skip global autoencoder")
	if (loc == 1):
		if (glo == 1):
			return ae_loc, X_train_hist, X_test_hist, ae_glo, X_train_global, X_test_global
		else:
			return ae_loc, X_train_hist, X_test_hist
	if (glo == 1):
		return ae_glo, X_train_global, X_test_global
