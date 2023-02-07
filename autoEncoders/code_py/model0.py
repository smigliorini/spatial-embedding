#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import myAutoencoder as enc

#
# Autoencoder
def auto_encoder(type,loc,glo,dimx,dimy,dimz,ldim_loc,ldim_glo,f1,f2,a,g):
	# type: 0=DENSE, 1=CNN, 2=CNN+DENSE, 3=CNN+DENSE(no class)
	# loc: 1=generate local autoencoder, 0=local autoencoder is not generated
	# glo: 1=generate global autoencoder, 0=global autoencoder is not generated
	# latent_dim is equal to the size of the embedding representation of the histogram for DENSE network
	# while for CNN it is equal to the number of values in each cell of the embedding, thus the size of
	# the embedding is obtained by the formula: dimx/4 * dimy/4 * latent_dim
	if (loc == 1):
		latent_dim = ldim_loc
		if (type == 0):
			print("Embedding dim: ", latent_dim)
			print("Inizilizing Autoencoder DENSE local...")
			ae_loc = enc.Autoencoder_local(f1,f2,latent_dim,dimx,dimy,dimz)
		elif (type == 1):
			#print("Embedding dim: ", latent_dim*dimx/4*dimy/4)
			print("Embedding dim: ", latent_dim)
			print("Inizilizing Autoencoder CNN local...")
			ae_loc = enc.AutoencoderCNN_local(latent_dim,dimx,dimy,dimz,f1,f2)
		elif (type == 2):
			print("Embedding dim: ", latent_dim)
			print("Inizilizing Autoencoder CNN+DENSE local...")
			ae_loc = enc.AutoencoderCNNDense_local(latent_dim,dimx,dimy,dimz,f1,f2)
		else:
			print("Embedding dim: ", latent_dim)
			print("Inizilizing Autoencoder CNN+DENSE local no class...")
			ae_loc = enc.AutoencoderCNNDense_local(latent_dim,dimx,dimy,dimz,f1,f2)
			ae_loc, encoder, decoder = enc.AutoencoderCNNDense_local_noClass(f1,f2,latent_dim,dimx,dimy,dimz)

		#
		ae_loc.compile(optimizer='adam', loss=losses.MeanSquaredError())
		# ae_loc.compile(optimizer='adam', loss=losses.MeanAbsolutePercentageError())
		# ae_loc.compile(optimizer='adam', loss='binary_crossentropy')

		# splitting train and test 0.2
		print("Splitting training and test set...")
		# local histograms
		X_train_hist, X_test_hist = train_test_split(a, test_size=0.2)
		# training
		print("Training Autoencoder local...")
		ae_loc.fit(X_train_hist, X_train_hist, batch_size=16, epochs=50, shuffle=True, validation_data=(X_test_hist, X_test_hist))
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
			ae_glo = enc.AutoencoderCNN_global(latent_dim,dimx,dimy)
		# ae_glo.compile(optimizer='adam', loss='binary_crossentropy')
		ae_glo.compile(optimizer='adam', loss=losses.MeanSquaredError())
		# ae_glo.compile(optimizer='adam', loss=losses.MeanAbsolutePercentageError())
		# splitting train and test 0.2
		print("Splitting training and test set...")
		# global histograms
		X_train_global, X_test_global = train_test_split(np.reshape(g,(g.shape[0],g.shape[1],g.shape[2],1)), test_size=0.2)
		# training
		print("Training Autoencoder global...")
		ae_glo.fit(X_train_global, X_train_global, batch_size=16, epochs=50, shuffle=True, validation_data=(X_test_global, X_test_global))
	else:
		print("Skip global autoencoder")
	if (loc == 1):
		if (glo == 1 and type < 3):
			return ae_loc, X_train_hist, X_test_hist, ae_glo, X_train_global, X_test_global
		elif (glo == 1 and type == 3):
			return ae_loc, encoder, decoder, X_train_hist, X_test_hist, ae_glo, X_train_global, X_test_global
		elif (glo == 0 and type == 3):
			return ae_loc, encoder, decoder, X_train_hist, X_test_hist
		else:
			return ae_loc, X_train_hist, X_test_hist
	if (glo == 1):
		return ae_glo, X_train_global, X_test_global
