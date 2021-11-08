#!/usr/bin/env python3
import model0 as m
a_new, g_new = m.gh.gen_input_from_file(128,128,6,'histograms/new_datasets','mbr/mbr_alldatasets.csv','')
a_large, g_large = m.gh.gen_input_from_file(128,128,6,'histograms/large_datasets','mbr/mbr_alldatasets.csv','')
a_medium, g_medium = m.gh.gen_input_from_file(128,128,6,'histograms/medium_datasets','mbr/mbr_alldatasets.csv','_m')
a_small, g_small = m.gh.gen_input_from_file(128,128,6,'histograms/small_datasets','mbr/mbr_alldatasets.csv','_s')
a_gap, g_gap = m.gh.gen_input_from_file(128,128,6,'histograms/gap_datasets','mbr/mbr_alldatasets.csv','')
#
a_new.shape
a_large.shape
a_medium.shape
a_small.shape
a_gap.shape
#
a_tot = m.np.concatenate((a_new,a_large,a_gap,a_medium,a_small))
g_tot = m.np.concatenate((g_new,g_large,g_gap,g_medium,g_small))
#
a_tot.shape
g_tot.shape
#
r_tot = m.np.zeros((2552,128,128))
b_tot = m.np.zeros((2552,))
# normalization
a_tot_norm_log = m.gh.nor_a(a_tot,1)
g_tot_norm_log = m.gh.nor_g(g_tot,1)
#
freq0 = m.gh.count_frequency(a_tot_norm_log[:,:,:,0], 128, 128)
import plot as p
p.plot_freq(freq0)
...
# from tensorflow import keras
# from sklearn.model_selection import train_test_split
# model = keras.models.load_model('path/to/location')
# autoenc_a_tot_log = keras.models.load_model('autoenc_a_tot_log')
# a_train_hist, a_test_hist = train_test_split(a_tot_norm_log, test_size=0.2)
#
# or
#
autoenc_a_tot_log, a_tot_train, a_tot_test =  m.auto_encoder(1,1,0,128,128,6,3,1,a_tot_log,g_tot_log,b_tot,r_tot)
enc_a_tot_test = autoenc_a_tot_log.encoder(a_tot_test).numpy()
dec_a_tot_test = autoenc_a_tot_log.decoder(enc_a_tot_test).numpy()
enc_a_tot_test_reshape = m.np.reshape(enc_a_tot_test, (-1,16,64,3))
p.plot_h6_mix_neg_emb(a_tot_test, dec_a_tot_test, enc_a_tot_test_reshape, 0, 10)
#
autoenc_g_tot_log, g_tot_train, g_tot_test =  m.auto_encoder(1,0,1,128,128,6,3,2,a_tot_norm_log,g_tot_norm_log,b_tot,r_tot)
enc_g_tot_test = autoenc_g_tot_log.encoder(g_tot_test).numpy()
dec_g_tot_test = autoenc_g_tot_log.decoder(enc_g_tot_test).numpy()
enc_g_tot_test_reshape = m.np.reshape(enc_g_tot_test, (-1,16,64,1))
