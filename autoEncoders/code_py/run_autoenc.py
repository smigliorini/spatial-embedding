#!/usr/bin/env python3
import model0 as m
import numpy as np
import time
import plot as p
import generate_histogram as gh
from sklearn.model_selection import train_test_split

#
# computes WMAPE real by considering the prediction of the model md
# Params:
# file_in_loc, file_in_glo: files containing the numpy array of the local and global histograms respectively 
#                           ('histograms_loc_tot.npy' and 'histograms_glo_tot.npy')
# min_a, max_a, min_g, max_g: min and max values for normalization
# md: autoEncorder in the class form with two submodels md.encoder and md.decorder
# rows, cols, z: dimension of the histograms
# filename: name of the file containiAng the plot of 10 actual and decoded histograms with embeddings
def real_test(file_in_loc,file_in_glo,min_a,max_a,min_g,max_g,md,rows,cols,z,filename):
	a_real = np.load(file_in_loc)
	g_real = np.load(file_in_glo)
	a_real_nor, min_real, max_real = m.gh.nor_g_ab(a_real,1,min_a,max_a)
	g_real_nor, min_real_g, max_real_g = m.gh.nor_g_ab(g_real,1,min_g,max_g)
	enc_a_real = md.encoder(a_real_nor).numpy()
	dec_a_real = md.decoder(enc_a_real).numpy()
	enc_a_real_reshape = np.reshape(enc_a_real, (-1,rows,cols,z))
	p.plot_h6_mix_emb(a_real_nor, dec_a_real, enc_a_real_reshape, 30, 10, filename, 1)
	dec_a_real_den = gh.denorm_g_ab(dec_a_real,1,min_a,max_a)
	wm , wmf = wmape(a_real,dec_a_real_den)
	print ('WMAPE real: ',wm, ' WMAPE features: ',wmf)

#
# computes WMAPE real by considering the prediction of the model md for Italy files
# Params: see real_test 
def it_test(min_a,max_a,min_g,max_g,md,rows,cols,filename):
	a_itt = np.load('histograms_loc_it.npy')
	g_itt = np.load('histograms_glo_it.npy')
	a_itt_nor, min_itt, max_itt = gh.nor_g_ab(a_itt,1,min_a,max_a)
	g_itt_nor, min_itt_g, max_itt_g = gh.nor_g_ab(g_itt,1,min_g,max_g)
	enc_a_test = md.encoder(a_itt_nor).numpy()
	dec_a_test = md.decoder(enc_a_test).numpy()
	enc_a_test_reshape = np.reshape(enc_a_test, (-1,rows,cols,3))
	p.plot_h6_mix_emb(a_itt_nor, dec_a_test, enc_a_test_reshape, 0, 2, filename ,0.5)

#
# Normalized and train a noClass autoencoder
# Load previously saved tensors containing dataset histograms, normalize histograms
# Create an autoencoder of type nonCLASS and trains it
# See also nor_and_train
def nor_and_train_noclass(file_a, file_test_a, file_g, file_test_g, latent_dim, loc, glo, f1, f2):
	# file_a: name of the file containing the whole set of data points or only the training set
        # file_test_a: 'XXX' if file_a contains the whole set of data points, the name of the file containing the test set otherwise
        # file_g and file_test_g: same as file_a and file_test_a for the global histograms
        # latent_dim of the embedding
        # loc: 1 for training a model for the generation of local embeddings, 0 otherwise
        # glo: 1 for training a model for the generation of global embeddings, 0 otherwise
        # f1, f2: the output of the dense net or the filters of the cnn net.

	# LOADING
	print("Loading files of histograms...")
	a_tot = np.load(file_a)
	g_tot = np.load(file_g)

	if (file_test_a == 'XXX'):
		# SPLITTING TRAINING and TEST set
		print('Splitting training (0.8) and testing sets (0.2)...')
		a_train, a_test, g_train, g_test = train_test_split(a_tot, g_tot, test_size=0.2)
	else:
		print('Loading test sets...')
		a_train = a_tot
		g_train = g_tot
		a_test = np.load(file_test_a)
		g_test = np.load(file_test_g)
	
	# NORMALIZING
	print('Normalizing histograms both training and test sets...')
	a_train, min_a, max_a = gh.nor_g_ab(a_train,1,-1,-1)
	g_train, min_g, max_g = gh.nor_g_ab(g_train,1,-1,-1)
	a_test, mint_a, maxt_a = gh.nor_g_ab(a_test,1,min_a,max_a)
	g_test, mint_g, maxt_g = gh.nor_g_ab(g_test,1,min_g,max_g)

	
	# CREATING AND TRAINING autoencoder (net_type always equal to 3)
	print('Training autoencoder...')
	autoenc, encoder, decoder, a_train_train, a_train_val =  m.auto_encoder(3,loc,glo,128,128,6,latent_dim,2,f1,f2,a_train,g_train)
	# autoenc.summary()
	
	# DELETE tensors
	# a_train = np.array([1, 2, 3])
	# g_train = np.array([4, 5, 6])
	a_tot = np.array([1, 2, 3])
	g_tot = np.array([4, 5, 6])
	print('Input tensors deleted!')

	# TESTING
	print('Testing autoencoder...')
	enc_a_test = encoder.predict(a_test)
	dec_a_test = decoder.predict(enc_a_test)
	print('encoder shape: ', enc_a_test.shape)
	print('decoder shape: ', dec_a_test.shape)
	
	# DENORMALIZING
	print('Denormalizing...')
	a_test_denor = gh.denorm_g_ab(a_test,1,min_a,max_a)
	a_train_denor = gh.denorm_g_ab(a_train,1,min_a,max_a)
	dec_a_test_denor = gh.denorm_g_ab(dec_a_test,1,min_a,max_a)
	
	# PLOTTING
	start_h = 30
	end_h = 40
	file_name = "ae_CNNDENSE_noClass_"
	file_name += str(f1) + "-" + str(f2) + "_" + "emb_" + str(latent_dim) + "_" + str(start_h) + "-" + str(end_h)
	print('Plotting...')
	p.plot_h6_mix_emb(a_test, dec_a_test, enc_a_test, start_h, (end_h - start_h), file_name, 0)
	wm, wmf = wmape(a_test_denor,dec_a_test_denor)
	print("WMAPE: ", wm, " WMAPE features: ", wmf)
	return autoenc, encoder, decoder, a_train_denor, a_test_denor, dec_a_test_denor, enc_a_test, min_a, max_a, min_g, max_g

# Normalized and train a class autoencoder
# Load previously saved tensors containing dataset histograms, normalize histograms
# Create an autoencoder of type CLASS and trains it
# See also nor_and_train
def nor_and_train(file_a, file_test_a, file_g, file_test_g, latent_dim, net_type, loc, glo, f1, f2):
	# file_a: name of the file containing the whole set of data points or only the training set
	# file_test_a: 'XXX' if file_a contains the whole set of data points, the test set otherwise
	# file_g and file_test_g: same as file_a and file_test_a
	# latent_dim of the embedding
	# net_type: = 0 DENSE, 1 CNN, 2 CNN+DENSE
	# loc: 1 for training a model for the generation of local embeddings, 0 otherwise
	# glo: 1 for training a model for the generation of global embeddings, 0 otherwise
	# f1, f2: the output of the dense net or the filters of the cnn net.
	#
	a_tot = np.load(file_a)
	g_tot = np.load(file_g)
	time_exe = np.zeros((3,), dtype=float)
	time0 = time.time()
	# normalization
	print('Normalizing histograms (total or training set)...')
	a_tot_norm_log, min_a, max_a = gh.nor_g_ab(a_tot,1,-1,-1)
	g_tot_norm_log, min_g, max_g = gh.nor_g_ab(g_tot,1,-1,-1)
	#
	#freq0 = gh.count_frequency(a_tot_norm_log[:,:,:,0], 128, 128)
	#freq3 = gh.count_frequency(a_tot_norm_log[:,:,:,3], 128, 128)
	#p.plot_freq(freq0)
	#p.plot_freq(freq3)
	#
	time_exe[0] = time.time()-time0
	print('Time for normalization of training set: ',time_exe[0])
	time0 = time.time()
	#
	if (file_test_a == 'XXX'):
		print('Splitting training and testing sets...')
		a_train, a_test, g_train, g_test = train_test_split(a_tot_norm_log, g_tot_norm_log, test_size=0.2)
	else:
		print('Loading test sets...')
		a_train = a_tot_norm_log
		g_train = g_tot_norm_log
		a_test = np.load(file_test_a)
		g_test = np.load(file_test_g)
		print('Normalizing histograms (test sets)...')
		a_test, mint_a, maxt_a = gh.nor_g_ab(a_test,1,min_a,max_a)	
		g_test, mint_g, maxt_g = gh.nor_g_ab(g_test,1,min_g,max_g)
	# print("A_TEST: ", a_test[0,0,0,0]," ",a_test[10,0,0,0])
	#
	#
	print('Training autoencoder...')
	# try: 256, 512, 768, 1536, 2304, 3072
	# latent_dim = 1536
	# DENSE, LOCAL YES, GLOBAL NO - AUTOENCODER = m.auto_encoder(0,1,0
	# CNN, LOCAL YES, GLOBAL NO - AUTOENCODER = m.auto_encoder(1,1,0
	# CNN+DENSE, LOCAL YES, GLOBAL NO - AUTOENCODER = m.auto_encoder(2,1,0
	autoenc, a_train_train, a_train_val =  m.auto_encoder(net_type,loc,glo,128,128,6,latent_dim,2,f1,f2,a_train,g_train)
	print("A_TEST: ", a_test[0,0,0,0]," ",a_test[10,0,0,0])
	time_exe[1] = time.time()-time0
	print('Time for training: ',time_exe[1])
	time0 = time.time()
	autoenc.summary()
	a_train = np.array([1, 2, 3])
	g_train = np.array([4, 5, 6])
	a_tot = np.array([1, 2, 3])
	g_tot = np.array([4, 5, 6])
	a_tot_norm_log = np.array([1, 2, 3])
	g_tot_norm_log = np.array([1, 2, 3])
	print('All arrays deleted!')
	enc_a_test = autoenc.encoder(a_test).numpy()
	dec_a_test = autoenc.decoder(enc_a_test).numpy()
	print('encoded: ', enc_a_test.shape)
	if (latent_dim == 512):
		enc_a_test_reshape = np.reshape(enc_a_test, (-1,16,16,2))
	elif (latent_dim == 256):
		enc_a_test_reshape = np.reshape(enc_a_test, (-1,16,16,1))
	else:
		enc_a_test_reshape = np.reshape(enc_a_test, (-1,16,int(latent_dim/(16*3)),3))
	print("A_TEST: ", a_test[0,0,0,0]," ",a_test[10,0,0,0])
	time_exe[2] = time.time()-time0
	print('Time for testing: ',time_exe[2])
	print('Times: ',time_exe)
	a_test_denor = gh.denorm_g_ab(a_test,1,min_a,max_a)
	dec_a_test_denor = gh.denorm_g_ab(dec_a_test,1,min_a,max_a)
	print("A_TEST: ", a_test[0,0,0,0]," ",a_test[10,0,0,0])
	print("A_TEST_DEN: ", a_test_denor[0,0,0,0]," ",a_test_denor[10,0,0,0])
	start_h = 30
	end_h = 40
	file_name = "ae_"
	if (net_type == 0):
		file_name += "DENSE_"
	elif (net_type == 1):
		file_name += "CNN_"
	else:
		file_name += "CNNDense_"
	file_name += str(f1) + "-" + str(f2) + "_" + "emb_" + str(latent_dim) + "_" + str(start_h) + "-" + str(end_h)
	print('Plotting...')
	enc_a_test_reshape, min_enc, max_enc = gh.nor_g_ab(enc_a_test_reshape,0,-1,-1)
	p.plot_h6_mix_emb(a_test, dec_a_test, enc_a_test_reshape, start_h, (end_h - start_h), file_name, 0)
	wm, wmf = wmape(a_test_denor,dec_a_test_denor)
	print("WMAPE: ", wm, " WMAPE features: ", wmf)
	return autoenc, a_test, dec_a_test, a_test_denor, dec_a_test_denor, enc_a_test_reshape, min_a, max_a, min_g, max_g
#
	#autoenc_g_tot_log, g_tot_train, g_tot_test =  m.auto_encoder(1,0,1,128,128,6,3,2,a_tot_norm_log,g_tot_norm_log,b_tot,r_tot)
	#enc_g_tot_test = autoenc_g_tot_log.encoder(g_tot_test).numpy()
	#dec_g_tot_test = autoenc_g_tot_log.decoder(enc_g_tot_test).numpy()
	#enc_g_tot_test_reshape = m.np.reshape(enc_g_tot_test, (-1,16,64,2))
	#p.plot_h6_mix_neg_emb_g(g_tot_test, dec_g_tot_test, enc_g_tot_test_reshape, 0, 10)

# Compute WMAPE between original array (orig) and predict array (dec)
#
def wmape(orig,dec):
	wmape_den_f = np.zeros((orig.shape[3],), dtype=float)
	wmape_num_f = np.zeros((orig.shape[3],), dtype=float)
	for i in range(orig.shape[0]):
		for x in range(orig.shape[1]):
			for y in range(orig.shape[2]):
				for f in range(orig.shape[3]):
					wmape_num_f[f] += abs(orig[i,x,y,f] - dec[i,x,y,f])
					wmape_den_f[f] += orig[i,x,y,f] 
	wmape_f = wmape_num_f/wmape_den_f
	wmape_num = 0.0
	for f in range(orig.shape[3]):
		wmape_num += wmape_f[f]
	wmape = wmape_num/orig.shape[3]
	return wmape, wmape_f
