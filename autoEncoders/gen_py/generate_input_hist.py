#!/usr/bin/env python3
import numpy as np
import time
import plot as p
import generate_histogram as gh

# Generation of random training set
# Param:
# n: number of histograms to be generated
# dimx,dimy,dims: dimensions of the histograms
# norm_type: 0 standardization, 1 normalization min-max
# distr: 0 uniform, 1 diagonal
# Returns:
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

# load HISTOGRAMS of real data regarding Italy into a numpy array (tensor)
# Param:
# file_out_loc: name of the output file containing the numpy array (128,128,6) representing the local histograms
# file_out_glo: name of the output file containing the numpy array (128,128,2) representing the global histograms
def load_hist_real_italy (file_out_loc, file_out_glo):
	a_it, g_it = gh.gen_input_from_file(128,128,6,'histograms/real_datasets/italy','mbr/mbr_italy.csv',0,'')
	print('Histograms Italy loaded!')
	np.save(file_out_loc,a_it)
	np.save(file_out_glo,g_it)

# load HISTOGRAMS of all synthetic datasets (small, medium, large, gap, new small, real) into a numpy array (tensor)
# Param:
# file_out_loc: name of the output file containing the numpy array (128,128,6) representing the local histograms
# file_out_glo: name of the output file containing the numpy array (128,128,2) representing the global histograms
def load_hist (file_out_loc, file_out_glo):
	time_exe = np.zeros((2,), dtype=float)
	print('Loading histograms...')
	time0 = time.time()
	a_real, g_real = gh.gen_input_from_file(128,128,6,'histograms/real_datasets','mbr/mbr_alldatasets.csv',0,'')
	a_new, g_new = gh.gen_input_from_file(128,128,6,'histograms/new_datasets','mbr/mbr_alldatasets.csv',0,'')
	a_large, g_large = gh.gen_input_from_file(128,128,6,'histograms/large_datasets','mbr/mbr_alldatasets.csv',0,'')
	a_gap, g_gap = gh.gen_input_from_file(128,128,6,'histograms/gap_datasets','mbr/mbr_alldatasets.csv',0,'')
	a_medium, g_medium = gh.gen_input_from_file(128,128,6,'histograms/medium_datasets','mbr/mbr_alldatasets.csv',0,'_m')
	a_small, g_small = gh.gen_input_from_file(128,128,6,'histograms/small_datasets','mbr/mbr_alldatasets.csv',0,'_s')
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
	a_tot = np.concatenate((a_real,a_new,a_large,a_gap,a_medium,a_small))
	g_tot = np.concatenate((g_real,g_new,g_large,g_gap,g_medium,g_small))
#
	a_tot.shape
	g_tot.shape
#
	np.save(file_out_loc,a_tot)
	np.save(file_out_glo,g_tot)
	print('Times: ',time_exe)

