#!/usr/bin/env python3
import model0 as m
import numpy as np
import time
import plot as p
from sklearn.model_selection import train_test_split
# load HISTOGRAMS real data
def load_hist_real ():
	a_it, g_it = m.gh.gen_input_from_file(128,128,6,'histograms/real_datasets/italy','mbr/mbr_italy.csv',0,'')
	print('Histograms Italy loaded!')
	m.np.save('histograms_loc_it',a_it)
	m.np.save('histograms_glo_it',g_it)
# load HISTOGRAMS
def load_hist ():
	time_exe = np.zeros((2,), dtype=float)
	print('Loading histograms...')
	time0 = time.time()
	a_real, g_real = m.gh.gen_input_from_file(128,128,6,'histograms/real_datasets','mbr/mbr_alldatasets.csv',0,'')
	a_new, g_new = m.gh.gen_input_from_file(128,128,6,'histograms/new_datasets','mbr/mbr_alldatasets.csv',0,'')
	a_large, g_large = m.gh.gen_input_from_file(128,128,6,'histograms/large_datasets','mbr/mbr_alldatasets.csv',0,'')
	a_medium, g_medium = m.gh.gen_input_from_file(128,128,6,'histograms/medium_datasets','mbr/mbr_alldatasets.csv',0,'_m')
	a_small, g_small = m.gh.gen_input_from_file(128,128,6,'histograms/small_datasets','mbr/mbr_alldatasets.csv',0,'_s')
	a_gap, g_gap = m.gh.gen_input_from_file(128,128,6,'histograms/gap_datasets','mbr/mbr_alldatasets.csv',0,'')
#
	time_exe[0] = time.time()-time0
	print('Time for creation of training set: ',time_exe[0])
	time0 = time.time()
#
	print('Histograms loaded!')
	a_new.shape
	a_large.shape
	a_medium.shape
	a_small.shape
	a_gap.shape
#
	print('Concatenating histograms...')
	a_tot = m.np.concatenate((a_real,a_new,a_large,a_gap,a_medium,a_small))
	g_tot = m.np.concatenate((g_real,g_new,g_large,g_gap,g_medium,g_small))
#
	a_tot.shape
	g_tot.shape
#
	m.np.save('histograms_loc_tot_real',a_tot)
	m.np.save('histograms_glo_tot_real',g_tot)
	print('Times: ',time_exe)
def nor_train(file_a, file_g, latent_dim, net_type, loc, glo, f1, f2):
	a_tot = m.np.load(file_a)
	g_tot = m.np.load(file_g)
	time_exe = np.zeros((3,), dtype=float)
	time0 = time.time()
	# normalization
	print('Normalizing histograms...')
	a_tot_norm_log, min_a, max_a = m.gh.nor_g_ab(a_tot,1,-1,-1)
	g_tot_norm_log, min_g, max_g = m.gh.nor_g_ab(g_tot,1,-1,-1)
	#
	freq0 = m.gh.count_frequency(a_tot_norm_log[:,:,:,0], 128, 128)
	p.plot_freq(freq0)
	#
	time_exe[0] = time.time()-time0
	print('Time for normalization of training set: ',time_exe[0])
	time0 = time.time()
	print('Splitting training and testing sets...')
	#
	a_train, a_test, g_train, g_test = train_test_split(a_tot_norm_log, g_tot_norm_log, test_size=0.2)
	print("A_TEST: ", a_test[0,0,0,0]," ",a_test[10,0,0,0])
	#
	#
	print('Training autoencoder...')
	# try:  48 96 192 384, 480, 768, 1536, 3072
	# latent_dim = 1536
	# DENSE, LOCAL YES, GLOBAL NO - AUTOENCODER = m.auto_encoder(0,1,0
	# CNN, LOCAL YES, GLOBAL NO - AUTOENCODER = m.auto_encoder(1,1,0
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
	enc_a_test_reshape = m.np.reshape(enc_a_test, (-1,16,int(latent_dim/(3*16)),3))
	print("A_TEST: ", a_test[0,0,0,0]," ",a_test[10,0,0,0])
	time_exe[2] = time.time()-time0
	print('Time for testing: ',time_exe[2])
	print('Times: ',time_exe)
	a_test_denor = m.gh.denorm_g_ab(a_test,1,min_a,max_a)
	dec_a_test_denor = m.gh.denorm_g_ab(dec_a_test,1,min_a,max_a)
	print("A_TEST: ", a_test[0,0,0,0]," ",a_test[10,0,0,0])
	print("A_TEST_DEN: ", a_test_denor[0,0,0,0]," ",a_test_denor[10,0,0,0])
	start_h = 0
	end_h = 10
	file_name = "ae_"
	if (net_type == 0):
		file_name += "DENSE_"
	else:
		file_name += "CNN_"
	file_name += str(f1) + "-" + str(f2) + "_" + "emb_" + str(latent_dim) + "_" + str(start_h) + "-" + str(end_h)
	print('Plotting...')
	p.plot_h6_mix_neg_emb(a_test, dec_a_test, enc_a_test_reshape, start_h, (end_h - start_h), file_name)
	return autoenc, a_test, dec_a_test, a_test_denor, dec_a_test_denor, enc_a_test_reshape, min_a, max_a, min_g, max_g
#
	#autoenc_g_tot_log, g_tot_train, g_tot_test =  m.auto_encoder(1,0,1,128,128,6,3,2,a_tot_norm_log,g_tot_norm_log,b_tot,r_tot)
	#enc_g_tot_test = autoenc_g_tot_log.encoder(g_tot_test).numpy()
	#dec_g_tot_test = autoenc_g_tot_log.decoder(enc_g_tot_test).numpy()
	#enc_g_tot_test_reshape = m.np.reshape(enc_g_tot_test, (-1,16,64,2))
	#p.plot_h6_mix_neg_emb_g(g_tot_test, dec_g_tot_test, enc_g_tot_test_reshape, 0, 10)
def wmape(orig,dec):
	wmape_den_f = m.np.zeros((orig.shape[3],), dtype=float)
	wmape_num_f = m.np.zeros((orig.shape[3],), dtype=float)
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