#!/usr/b/in/env python3
import os
from os import listdir, mkdir, path, sep
from os.path import isfile, join
from tensorflow import keras
import numpy as np
import random as rd
import math
import csv
import generate_histogram as gh
import plot as p

# CONST -------------------------
#
DIM_H_X = 128
DIM_H_Y = 128
DIM_H_Z = 6

DIM_HG_Z = 1

# emb3: latent_dim = 1536
#DIM_E_X = 32
#DIM_E_Y = 16
#DIM_E_Z = 3

# emb0: latent_dim = 3072
DIM_E_X = 32
DIM_E_Y = 32
DIM_E_Z = 3

DIM_EG_Z = 2

NORM_MIN = [0. ,0. ,0. ,0. ,0. ,0.]
# only synthetic
NORM_MAX = [8.77805800e+06, 3.05404802e+09, 1.53571255e+08, 3.03019291e-02, 1.91233400e-01, 2.20753674e-01]
# synthetic + real
#NORM_MAX = [8.77805800e+06, 3.05404802e+09, 1.53571255e+08, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00]


NORM_MIN_G = 0.0
NORM_MAX_G = 8708693.144550692

X_MIN_REF = 0
X_MAX_REF = 10
Y_MIN_REF = 0
Y_MAX_REF = 10

MIN_CARD = 1

# -------------------------------

def get_embedding(local_enc, global_enc, rq_hist, datasetFile, mbr):
    # get local histogram
    hist_local = gh.gen_hist_from_file(DIM_H_X,DIM_H_Y,DIM_H_Z,datasetFile)
    hist_local_norm, min_a, max_a = gh.nor_g_ab(hist_local.reshape((1,DIM_H_X,DIM_H_Y,DIM_H_Z)), 1, NORM_MIN, NORM_MAX)
    emb_local = local_enc.encoder(hist_local_norm.reshape((1,DIM_H_X,DIM_H_Y,DIM_H_Z)))

    # computing global histogram
    hist_glob = gh.gen_global_hist(hist_local, DIM_H_X, DIM_H_Y, mbr)
    hist_glob_norm, min_g, max_g = gh.nor_g_ab(hist_glob.reshape((1,DIM_H_X,DIM_H_Y)), 1, NORM_MIN_G, NORM_MAX_G)
    emb_global = global_enc.encoder(hist_glob_norm.reshape((1,DIM_H_X,DIM_H_Y,DIM_HG_Z)))

    # computing embedding of rq_histogram
    emb_rq = global_enc.encoder(rq_hist.reshape((1, DIM_H_X, DIM_H_Y, DIM_HG_Z)))

    return emb_local, emb_global, emb_rq

def cutting (file_a_x, file_a_x1, file_a_y, file_a_ds, from_x, to_x, suff):
	x_a = np.load(file_a_x)
	x1_a = np.load(file_a_x1)
	y_a = np.load(file_a_y)
	ds_a = np.load(file_a_ds)
	x_i = np.zeros((x_a.shape[0]))
	for i in range(from_x,to_x):
		x_i[i] = 1

	x_a_c = x_a[x_i==1]
	x1_a_c = x1_a[x_i==1]
	y_a_c = y_a[x_i==1]
	ds_a_c = ds_a[x_i==1]

	dim = to_x - from_x
	np.save("x_"+str(dim)+suff, x_a_c)
	np.save("x1_"+str(dim)+suff, x1_a_c)
	np.save("y_"+str(dim)+suff, y_a_c)
	np.save("ds_"+str(dim)+suff, ds_a_c)

def unioning_index (file_a_x, file_b_x, file_s_a, file_s_b):
	x_a = np.load(file_a_x)
	x_b = np.load(file_b_x)
	s_a = np.load(file_s_a)
	s_b = np.load(file_s_b)

	num_a = np.sum(s_a)
	num_b = np.sum(s_b)
	x_res = np.zeros((num_a+num_b,x_a.shape[1],x_a.shape[2],x_a.shape[3]), dtype=float)
	
	j = 0
	for i in range(x_a.shape[0]):
		if (s_a[i] == 1):
			x_res[j] = x_a[i]
			j += 1
	for i in range(x_b.shape[0]):
		if (s_b[i] == 1):
			x_res[j] = x_b[i]
			j += 1
		
	np.save("x_"+str(j)+"_union",x_res)

def unioning (mode, file_a_x, file_a_x1, file_a_y, file_a_ds, file_b_x, file_b_x1, file_b_y, file_b_ds, num_a, num_b):
	# mode: 0 generates values in y, 1 generate classes
	# classed are defined by the following thresholds
	th = np.array([0.001, 0.01, 0.1, 0.15, 0.2, 0.3, 0.4, 0.55, 0.7, 0.9, 1.0], dtype=float)

	x_a = np.load(file_a_x)
	x1_a = np.load(file_a_x1)
	y_a = np.load(file_a_y)
	ds_a = np.load(file_a_ds)

	x_b = np.load(file_b_x)                
	x1_b = np.load(file_b_x1)
	y_b = np.load(file_b_y)
	ds_b = np.load(file_b_ds)

	seed_a = np.zeros((x_a.shape[0]) , dtype=int)
	seed_b = np.zeros((x_b.shape[0]), dtype=int)
	x_res = np.zeros((num_a+num_b,x_a.shape[1],x_a.shape[2],x_a.shape[3]), dtype=float)
	x1_res = np.zeros((num_a+num_b,x1_a.shape[1]), dtype=float)
	ds_res = np.empty((num_a+num_b,2), dtype='<U30')
	if (mode == 1):
		y_res = np.zeros((num_a+num_b), dtype=int)
	else:
		y_res = np.zeros((num_a+num_b), dtype=float)
	p_a = (num_a*1.1) / x_a.shape[0]
	p_b = (num_b*1.1) / x_b.shape[0]

	num_chosen = 0
	for i in range(x_a.shape[0]):
		if (rd.random() <= p_a):
			seed_a[i] = 1
			x_res[num_chosen] = x_a[i]
			x1_res[num_chosen] = x1_a[i]
			ds_res[num_chosen] = ds_a[i]
			if (mode == 1):
				if (y_a[i] < th[0]):
					y_res[num_chosen] = 0
				elif (y_a[i] < th[1]):
					y_res[num_chosen] = 1
				elif (y_a[i] < th[2]):
					y_res[num_chosen] = 2
				elif (y_a[i] < th[3]):
					y_res[num_chosen] = 3
				elif (y_a[i] < th[4]):
					y_res[num_chosen] = 4
				elif (y_a[i] < th[5]):
					y_res[num_chosen] = 5
				elif (y_a[i] < th[6]):
					y_res[num_chosen] = 6
				elif (y_a[i] < th[7]):
					y_res[num_chosen] = 7
				elif (y_a[i] < th[8]):
					y_res[num_chosen] = 8
				elif (y_a[i] < th[9]):
					y_res[num_chosen] = 9
				else:
					y_res[num_chosen] = 10
			else:
				y_res[num_chosen] = y_a[i]
			num_chosen += 1
			if (num_chosen == num_a):
				break
	na = num_chosen
	num_chosen = 0
	for i in range(x_b.shape[0]):
		if (rd.random() <= p_b):
			seed_b[i] = 1
			x_res[na+num_chosen] = x_b[i]
			x1_res[na+num_chosen] = x1_b[i]
			ds_res[na+num_chosen] = ds_b[i]
			if (mode == 1):
				if (y_b[i] < th[0]):
					y_res[na+num_chosen] = 0
				elif (y_b[i] < th[1]):
					y_res[na+num_chosen] = 1
				elif (y_b[i] < th[2]):
					y_res[na+num_chosen] = 2
				elif (y_b[i] < th[3]):
					y_res[na+num_chosen] = 3
				elif (y_b[i] < th[4]):
					y_res[na+num_chosen] = 4
				elif (y_b[i] < th[5]):
					y_res[na+num_chosen] = 5
				elif (y_b[i] < th[6]):
					y_res[na+num_chosen] = 6
				elif (y_b[i] < th[7]):
					y_res[na+num_chosen] = 7
				elif (y_b[i] < th[8]):
					y_res[na+num_chosen] = 8
				elif (y_b[i] < th[9]):
					y_res[na+num_chosen] = 9
				else:
					y_res[na+num_chosen] = 10
			else:
				y_res[na+num_chosen] = y_b[i]

			num_chosen += 1
			if (num_chosen == num_b):
				break

	np.save("x_"+str(na+num_chosen)+"_union",x_res)
	np.save("x1_"+str(na+num_chosen)+"_union",x1_res)
	np.save("y_"+str(na+num_chosen)+"_union",y_res)
	np.save("ds_"+str(na+num_chosen)+"_union",ds_res)
	return seed_a, seed_b

def gen_inputs_embs (new, mode, flag_sel_card, from_x, to_x, file_rq, file_mbr, file_result, file_summary, path_hist, flag_num_query, delim, real, emb, perc):
# new: 1 read one file of results, 0 many files are used (first version)
# flag_sel_card: 0 stores in y selectivity, 
#                1 cardinality, 
#                2 mbrTests,
#                3 mbrTests selectivity
# mode: 0 the embedding of rq is computed and added to x,
#       1 the glob_histogram of rq 32x32x1 is computed and added to x
#       2 x contains only the local embedding
#	3 only y is generated
#	file_rq: file containing the rectangles of query 'rq/rq_newDatasets.csv'
#	file_mbr: file containing the mbr of the datasets 'mbr/mbr_alldatasets.csv'
#	file_result: file containing the results of the queries 'rq/rq_result.csv'
#	file_summary: file containing the characteristics of the datasets 'dataset-summaries.csv'
#	path_hist: path where are stored the histograms 'histograms/new_datasets/' 
#	flag_num_querY: 0 use a counter, 1 use the numQuery in the file
# 	real: 1 if the results comes from a experiments with real data (different names for csv columns)
#	emb: autoencoder to be used for generating the embeddings; 0=emb0, 1=emb1, ...
#	perc: percentage of the total data points that are generated in the result files

	# LOCAL embeddings
	# decomment the models you want to use for embeddings generation, 
	# change also the file name for preserving previous generated files
	# and the DIM_E_X, ...

	if (real == 0):
		# embedding DENSE synthetic only
		f_emb0 = 'model/autoencoder_3072_CNNDense_newDatasets_SMLG'
		f_emb1 = 'model/autoencoder_DENSE3L_1024-512_emb384_synthetic'
		f_emb2 = 'model/autoencoder_DENSE3L_1024-512_emb1536_synthetic'
		# embedding CNN synthetic only
		f_emb3 = 'model/autoencoder_CNN_128-64_emb768_synthetic'
		f_emb4 = 'model/autoencoder_CNN_64-32_emb1536_synthetic'
	else:
		# embedding DENSE synthetic and real
		f_emb1 = 'model/autoencoder_DENSE3L_16-32_emb384_real'
		f_emb2 = 'model/autoencoder_DENSE3L_16-32_emb48_real'
		# embedding CNN synthetic and real
		f_emb3 = 'model/autoencoder_CNN3L_128-64_emb1536_real'
		f_emb4 = 'model/autoencoder_CNN3L_64-32_emb768_real'
	
	# generating 4 input datasets using all encoders
	global DIM_E_X
	global DIM_E_Y
	global DIM_E_Z
	global NORM_MAX
	if (real == 0):
		print("emb trained with synthetic data")
		if (emb == 0):
			# emb0: latent_dim = 3072
			emb_txt = "emb0"
			DIM_E_X = 32
			DIM_E_Y = 32
			DIM_E_Z = 3
			e = keras.models.load_model(f_emb0)
		elif (emb == 1):
			# emb1: latent_dim = 384
			emb_txt = "emb1"
			DIM_E_X = 16
			DIM_E_Y = 8
			DIM_E_Z = 3
			e = keras.models.load_model(f_emb1)
		elif (emb == 2):
			# emb2: latent_dim = 1536
			emb_txt = "emb2"
			DIM_E_X = 32
			DIM_E_Y = 16
			DIM_E_Z = 3
			e = keras.models.load_model(f_emb2)
		elif (emb == 3):
			# emb3: latent_dim = 768
			emb_txt = "emb3"
			DIM_E_X = 16
			DIM_E_Y = 16
			DIM_E_Z = 3
			e = keras.models.load_model(f_emb3)
		else:
			# emb4: latent_dim = 1536
			emb_txt = "emb4"
			DIM_E_X = 32
			DIM_E_Y = 16
			DIM_E_Z = 3
			e = keras.models.load_model(f_emb4)
	else:
		NORM_MAX = [8.77805800e+06, 3.05404802e+09, 1.53571255e+08, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00]
		print("emb trained with synthetic+real data")
		if (emb == 1):
			# emb1: latent_dim = 384
			emb_txt = "emb1_real"
			DIM_E_X = 16
			DIM_E_Y = 8
			DIM_E_Z = 3
			e = keras.models.load_model(f_emb1)
		elif (emb == 2):
			# emb2: latent_dim = 48
			emb_txt = "emb2_real"
			DIM_E_X = 4
			DIM_E_Y = 4
			DIM_E_Z = 3
			e = keras.models.load_model(f_emb2)
		elif (emb == 3):
			# emb3: latent_dim = 1536
			emb_txt = "emb3_real"
			DIM_E_X = 32
			DIM_E_Y = 16
			DIM_E_Z = 3
			e = keras.models.load_model(f_emb3)
		else:
			# emb4: latent_dim = 768
			emb_txt = "emb4_real"
			DIM_E_X = 16
			DIM_E_Y = 16
			DIM_E_Z = 3
			e = keras.models.load_model(f_emb4)
	print(emb_txt)
			
	# GLOBAL embeddings
	# one model
	f_gemb0 = 'model/model_2048_CNNDense_newDatasets_SMLG_global_new'
	g0 = keras.models.load_model(f_gemb0)

	num = to_x - from_x

	if (new == 0):
		out_x, out_y, out_hist_rq, out_x1, out_name_distr  = gen_rq_input_from_file(mode, flag_sel_card, from_x, to_x, e, g0, file_rq, flag_num_query, file_mbr, file_result, file_summary, path_hist, delim, real)
	else:
		out_x, out_y, out_x1, out_name_distr  = gen_rq_input_from_file_new(mode, flag_sel_card, from_x, to_x, e, g0, file_result, path_hist, delim, perc)
		
	np.save("ds_"+str(out_name_distr.shape[0])+"_rq_"+str(flag_sel_card),out_name_distr)
	if (mode == 3):
		np.save("y_"+str(out_y.shape[0])+"_rq_"+str(flag_sel_card),out_y)
		return
	if (mode == 2):
		np.save("y_"+str(out_y.shape[0])+"_rq_"+str(flag_sel_card),out_y)
		np.save("x1_"+str(out_x1.shape[0])+"_rq_"+str(mode),out_x1)
	np.save("x_"+str(out_x.shape[0])+"_rq_"+str(mode)+"_"+emb_txt,out_x)


def gen_res_file (file_in, file_res, file_out, delim):
# generate a result file starting from a file with the new format and the file produced by Ahmed
	rq_res = {}
	# Reading Result file
	with open(file_res, mode='r') as csv_file:
		csv_reader = csv.DictReader(csv_file, delimiter=delim)
		line_count = 0
		for row in csv_reader:
			name = row["dataset"]+"-"+row["numQuery"]
			rq_res[name] = dict([('cardinality', float(row["cardinality"])), ('mbrTests', float(row["mbrTests"]))])
			line_count += 1
			print("Line res: ", line_count)
	# Reading Input file
	nq = {}
	rq_out = {}
	with open(file_in, mode='r') as csv_file:
		csv_reader = csv.DictReader(csv_file, delimiter=',')
		line_count = 0
		for row in csv_reader:
			numq = 1
			find = 0
			for x in nq:
				if (x == row["dataset"]):
					numq = nq[x]
					nq[x] = numq + 1
					find = 1
					break
			if (find == 0):
				nq[row["dataset"]] = 2

			rq_to_search = row["dataset"]+"-"+str(numq)
			print("rq_to_search: ", rq_to_search)
			find_rq_res = 0
			for rr in rq_res:
				if (rr == rq_to_search):
					find_rq_res = 1
					break
			if (find_rq_res == 0):
				continue
			r = rq_res[rq_to_search]
			print("find rq res: ", r)
			row_out = row
			row_out["rq_card_real"] = r["cardinality"]
			row_out["rq_sel_real"] = r["cardinality"]/float(row["card"])
			row_out["mbrTests"] = r["mbrTests"]

			rq_out[rq_to_search] = row_out
			line_count += 1
			print("Line: ", line_count)
	 
	fieldnames = ['dataset', 'distr', 'card', 'minx', 'miny', 'maxx', 'maxy', 'rq_minx', 'rq_miny', 'rq_maxx', 'rq_maxy', 'rq_card', 'rq_sel', 'rq_card_real', 'rq_sel_real', 'mbrTests']
	with open(file_out, 'w', encoding='UTF8', newline='') as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		for i in rq_out:
			writer.writerow(rq_out[i])			

def gen_rq_input_from_file_hist (mode, flag_sel_card, from_x, to_x, resultFile, pathHist, delim, perc):
# mode: 2 x containg the histograms, x1 contains the MBR of the dataset and the query rectangle
#       3 only y is generated
#       -1 only distr is generated
# flag_sel_card: 0 stores in y the selectivity, 1 the cardinality, 2 the mbrTests, 3 mbrTests selectivity
# from_x, to_x: seleziona un sottoinsieme di data points
# resultFile: file containing the range query results and all necessary features of the dataset and query
#

        total = to_x - from_x
        total = int(total * (perc*1.05))
        out_y = np.zeros((total))
        out_name_distr = np.empty((total,2), dtype='<U40')
        if (mode == 2):
            out_x = np.zeros((total,DIM_H_X,DIM_H_Y,DIM_H_Z))
            out_x1 = np.zeros((total, 8))
        else:
            out_x = np.zeros((total,))
            out_x1 = np.zeros((total,))

        # removed since it is too costly to do
        #keep = np.ones((total), dtype=np.int8)

        # Reading Result file
        with open(resultFile, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file,delimiter=delim)
            line_count = 0
            for row in csv_reader:
                if (line_count == 0):
                    print(f'Column names are: {", ".join(row)}')
                if (line_count < from_x):
                    line_count += 1
                    continue
                if (line_count == to_x):
                    break

                if (rd.random() > perc):
                   continue

                # skip datasets too small
                # removed since it is too costly to do
                #if (float(row["card"]) <= MIN_CARD):
                #    keep[line_count - from_x] = 0

                # computing distribution
                out_name_distr[line_count - from_x] = [row["dataset"],row["distr"]]

                # if only distribution has to be generated continue to the next row
                if (mode == -1):
                    line_count += 1
                    continue

                fileHist = pathHist + row["dataset"] + "_summary.csv"
                rq0 = dict([('minx', float(row["rq_minx"])), ('miny', float(row["rq_miny"])), ('maxx', float(row["rq_maxx"])), ('maxy', float(row["rq_maxy"]))])
                mbr0 = dict([('minx', float(row["minx"])), ('miny', float(row["miny"])), ('maxx', float(row["maxx"])), ('maxy', float(row["maxy"]))])

                # computing X in the different cases
                if (mode == 2):
                    hist_local = gh.gen_hist_from_file(DIM_H_X,DIM_H_Y,DIM_H_Z, fileHist)
                    # IMPORTANT: HERE THE NORMALIZATION OF THE HISTOGRAM IS APPLIED: rememeber to check the parameters
                    hist_local_norm, min_a, max_a = gh.nor_g_ab(hist_local.reshape((1,DIM_H_X,DIM_H_Y,DIM_H_Z)), 1, NORM_MIN, NORM_MAX)
                    out_x[line_count - from_x] = hist_local_norm

                    x1 = np.zeros((8))
                    x1[0] = row["minx"]
                    x1[1] = row["miny"]
                    x1[2] = row["maxx"]
                    x1[3] = row["maxy"]
                    x1[4] = row["rq_minx"]
                    x1[5] = row["rq_miny"]
                    x1[6] = row["rq_maxx"]
                    x1[7] = row["rq_maxy"]
                    out_x1[line_count - from_x] = x1
                else:
                    # in the other cases x and x1 are useless
                    x = 0.0
                    x1 = 0.0

                if (flag_sel_card == 0):
                    out_y[line_count-from_x] = float(row["rq_sel_real"])
                else:
                    if (flag_sel_card == 1):
                        out_y[line_count-from_x] = float(row["rq_card_real"])
                    elif (flag_sel_card == 2):
                        out_y[line_count-from_x] = float(row["mbrTests"])
                    else:
                        out_y[line_count-from_x] = float(row["mbrTests"]/float(row["card"]))

                line_count += 1
                if (line_count % 100 == 0):
                    print("line: ", str(line_count))

            # filter out the data points that violate constraints
            # removed since it is too costly to do
            #print("Discarding small datasets...")
            #out_x = out_x[keep==1]
            #out_y = out_y[keep==1]
            #out_x1 = out_x1[keep==1]
            #out_name_distr = out_name_distr[keep==1]

            print("Salvataggio su file out_x")
            np.save("x_rq_"+str(out_x.shape[0]), out_x)
            print("Salvataggio su file out_x1")
            np.save("x1_rq_"+str(out_x1.shape[0]), out_x1)
            print("Salvataggio su file out_y")
            np.save("y_rq_"+str(out_y.shape[0]), out_y)
            print("Salvataggio su file out_name_distr")
            np.save("ds_rq_"+str(out_name_distr.shape[0]), out_name_distr)
            return 1

def gen_rq_input_from_file_new (mode, flag_sel_card, from_x, to_x, local_enc, global_enc, resultFile, pathHist, delim, perc):
# mode: 0 the embedding of rq is computed and added to x,
#       1 the glob_histogram of rq 32x32x1 is computed and added to x
#       2 x contains only the local embedding
#       3 only y is generated
#       -1 only distr is generated
# flag_sel_card: 0 stores in y the selectivity, 1 the cardinality, 2 the mbrTests, 3 mbrTests selectivity
# from_x, to_x: seleziona un sottoinsieme di data points
# local_enc: encoder for local embedding
# global_enc: encorder for global embedding
# resultFile: file containing the range query results and all necessary features of the dataset and query
#

        total = to_x - from_x
        if (perc < 1.0):	
            total = int(total * (perc*1.01))
        out_y = np.zeros((total))
        out_hist_rq = np.zeros((total,DIM_H_X,DIM_H_Y,1))
        out_name_distr = np.empty((total,2), dtype='<U40')

        if (mode == 0):
            out_x = np.zeros((total,DIM_E_X,DIM_E_Y,(DIM_E_Z + 2*DIM_EG_Z)))
            out_x1 = np.zeros((1))
        elif (mode == 1):
            out_x = np.zeros((total,DIM_E_X,DIM_E_Y,(DIM_E_Z + DIM_EG_Z + 1)))
            out_x1 = np.zeros((1))
        elif (mode == 2):
            out_x = np.zeros((total,DIM_E_X,DIM_E_Y,DIM_E_Z))
            out_x1 = np.zeros((total, 8))
        else:
            out_x = np.zeros((total,))
            out_x1 = np.zeros((total,))

        keep = np.ones((total), dtype=np.int8)

        # Reading Result file
        with open(resultFile, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file,delimiter=delim)
            line_count = 0
            for row in csv_reader:
                if (line_count == 0):
                    print(f'Column names are: {", ".join(row)}')
                if (line_count < from_x):
                    line_count += 1
                    continue;
		# before was: line_count == to_x
                if (line_count == total):
                    break

                if (rd.random() > perc):
                    continue

                # skip datasets too small
                if (float(row["card"]) <= MIN_CARD):
                    keep[line_count - from_x] = 0

                # computing distribution
                out_name_distr[line_count - from_x] = [row["dataset"],row["distr"]]

                # if only distribution has to be generated continue to the next row
                if (mode == -1):
                    line_count += 1
                    continue

                # fileHist = pathHist + row["dataset"][0:16] + "s" + row["dataset"][16:] + "_summary.csv"
                fileHist = pathHist + row["dataset"] + "_summary.csv"
                rq0 = dict([('minx', float(row["rq_minx"])), ('miny', float(row["rq_miny"])), ('maxx', float(row["rq_maxx"])), ('maxy', float(row["rq_maxy"]))])
                mbr0 = dict([('minx', float(row["minx"])), ('miny', float(row["miny"])), ('maxx', float(row["maxx"])), ('maxy', float(row["maxy"]))])

		# computing X in the different cases
                # mode == 0: local emb, global emb, rq emb
                if (mode == 0):
                    hist_RQ = gen_rq_layer(rq0, DIM_H_X, DIM_H_Y)
                    embL, embG, embRQ = get_embedding(local_enc, global_enc, hist_RQ, fileHist, mbr0)
                    embL = embL.numpy().reshape((DIM_E_X, DIM_E_Y, DIM_E_Z))
                    embG = embG.numpy().reshape((32, 32, 2))
                    embRQ = embRQ.numpy().reshape((32, 32, 2))
                    x = np.concatenate([embL, embG, embRQ], axis=2)
                # mode == 1: local emb, global emb, rq hist
                elif (mode == 1):
                    hist_RQ = gen_rq_layer(rq0, DIM_H_X, DIM_H_Y)
                    # la chiamata di get_embedding non cambia ma embRQ viene buttato in questo caso!
                    embL, embG, embRQ = get_embedding(local_enc, global_enc, hist_RQ , fileHist, mbr0)
                    embL = embL.numpy().reshape((DIM_E_X, DIM_E_Y, DIM_E_Z))
                    embG = embG.numpy().reshape((32, 32, 2))
                    embRQ = 0
                    x = np.concatenate([embL, embG, hist_RQ], axis=2)
                # mode == 2: local emb and in a separate array global mbr and rq mbr
                elif (mode == 2):
                    h_zero = np.zeros((DIM_H_X,DIM_H_Y))
                    # la chiamata di get_embedding non cambia ma embRQ viene buttato in questo caso!
                    embL, embG, embRQ = get_embedding(local_enc, global_enc, h_zero , fileHist, mbr0)
                    embL = embL.numpy().reshape((DIM_E_X, DIM_E_Y, DIM_E_Z))
                    embRQ = 0
                    embG = 0
                    x = embL
                    x1 = np.zeros((8))
                    x1[0] = row["minx"]
                    x1[1] = row["miny"]
                    x1[2] = row["maxx"]
                    x1[3] = row["maxy"]
                    x1[4] = row["rq_minx"]
                    x1[5] = row["rq_miny"]
                    x1[6] = row["rq_maxx"]
                    x1[7] = row["rq_maxy"]
                    out_x1[line_count - from_x] = x1
                else:
                    # in the other cases x and x1 are useless
                    x = 0.0
                    x1 = 0.0

                out_x[line_count - from_x] = x

		# rq_sel o rq_sel_real
		# rq_card o rq_card_real	

                if (flag_sel_card == 0):
                    out_y[line_count-from_x] = float(row["rq_sel_real"])
                else:
                    if (flag_sel_card == 1):
                        out_y[line_count-from_x] = float(row["rq_card_real"])
                    elif (flag_sel_card == 2):
                        out_y[line_count-from_x] = float(row["mbrTests"])
                    else:
                        out_y[line_count-from_x] = float(row["mbrTests"]/float(row["card"]))

                line_count += 1
                if (line_count % 100 == 0):
                    print("line: ", str(line_count))

            # filter out the data points that violate constraints
            out_x = out_x[keep==1]
            out_y = out_y[keep==1]
            out_x1 = out_x1[keep==1]
            out_name_distr = out_name_distr[keep==1]

            return out_x, out_y, out_x1, out_name_distr

def gen_rq_input_from_file(mode, flag_sel_card, from_x, to_x, local_enc, global_enc, rqFile, flag_num_query, mbrFile, resultFile, cardFile, pathHist, delim, real):
# flag_sel_card: 0 stores in y the selectivity, 1 the cardinality, 2 the mbrTests, 3 mbrTests selectivity
# flag_num_querY: 0 use a counter, 1 use the numQuery in the file
# mode: 0 the embedding of rq is computed and added to x, 
#       1 the glob_histogram of rq 32x32x1 is computed and added to x
#       2 x contains only the local embedding 
#       3 only y is generated
#	-1 only distr is generated
# local_enc: encoder for local embedding
# global_enc: encorder for global embedding
# rqFile: the name of the file containing the MBR of the queries
#
        # Reading RQ file
        rq = {}
        with open(rqFile, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter=',')
            line_count = 0
            rq_count = 0
            dName_old = ""
            for row1 in csv_reader:
                if (line_count == 0):
                    print(f'Column names are: {", ".join(row1)}')
                    dName_old = row1["datasetName"]
               	print(f'\t{row1["datasetName"]},{row1["numQuery"]}: {row1["minX"]}, {row1["minY"]}, {row1["maxX"]}, {row1["maxY"]}.')
                dName = row1["datasetName"]
                if (dName != dName_old):
                    rq_count = 0
                if (flag_num_query == 0):
                    name = dName + "-" + str(rq_count)
                else:
                    name = dName + "-" + row1["numQuery"]
                print(name)
                rq[name] = dict([('minx', float(row1["minX"])), ('miny', float(row1["minY"])), ('maxx', float(row1["maxX"])),
                                 ('maxy', float(row1["maxY"]))])
                rq_count += 1
                dName_old = dName
                line_count += 1

        # Reading MBR file
        mbr = {}
        with open(mbrFile, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file,delimiter=',')
            line_count = 0
            for row in csv_reader:
                if (line_count == 0):
                    print(f'Column names are: {", ".join(row)}')
                if (real == 0):
                    print(f'\t{row["datasetName"]}: {row["minX"]}, {row["minY"]}, {row["maxX"]}, {row["maxY"]}.')
                else:
                    print(f'\t{row["datasetName"]}: {row["x1"]}, {row["y1"]}, {row["x2"]}, {row["y2"]}.')
                name = row["datasetName"]
                if (real == 0):
                    mbr[name] = dict([('minx', float(row["minX"])), ('miny', float(row["minY"])), ('maxx', float(row["maxX"])), ('maxy', float(row["maxY"]))])
                else:
                    mbr[name] = dict([('minx', float(row["x1"])), ('miny', float(row["y1"])), ('maxx', float(row["x2"])), ('maxy', float(row["y2"]))])
                line_count += 1

        # Reading Cardinality of datasets
        card = {}
        with open(cardFile, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter=delim)
            line_count = 0
            for row in csv_reader:
                if (line_count == 0):
                    print(f'Column names are: {", ".join(row)}')
                print(f'\t{row["datasetName"]}: {row["num_features"]}.')
                name = row["datasetName"]
                card[name] = dict([('numFeatures', float(row["num_features"])), ('distr', row["distribution"])])
                line_count += 1

        # Reading Result file
        #with open(resultFile, mode='r') as csv_file:
        #    csv_reader = csv.DictReader(csv_file,delimiter=delim)
        #    line_count = 0
        #    for row in csv_reader:
        #        line_count += 1

        total = to_x - from_x
        out_y = np.zeros((total))
        out_hist_rq = np.zeros((total,DIM_H_X,DIM_H_Y,1))
        out_name_distr = np.empty((total,2), dtype='<U40')

        if (mode == 0):
            out_x = np.zeros((total,DIM_E_X,DIM_E_Y,(DIM_E_Z + 2*DIM_EG_Z)))
            out_x1 = np.zeros((1))
        elif (mode == 1):
            out_x = np.zeros((total,DIM_E_X,DIM_E_Y,(DIM_E_Z + DIM_EG_Z + 1)))
            out_x1 = np.zeros((1))
        elif (mode == 2):
            out_x = np.zeros((total,DIM_E_X,DIM_E_Y,DIM_E_Z))
            out_x1 = np.zeros((total, 8))
        else:
            out_x = np.zeros((total,))
            out_x1 = np.zeros((total,))
	
        keep = np.ones((total), dtype=np.int8)
        out_rq = {}
        #out_hist = np.zeros((total,DIM_H_X,DIM_H_Y))
        with open(resultFile, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file,delimiter=delim)
            line_count = 0
            count = 0
            max_selectivity = 0.5
            uniform_distr = 0
            for row in csv_reader:
                if (line_count == 0):
                    print(f'Column names are: {", ".join(row)}')
                if (line_count == to_x):
                    break
                if (line_count < from_x):
                    line_count += 1
                    continue;

                c = card[row["dataset"]]
                if (c["numFeatures"] <= MIN_CARD):
                    keep[line_count - from_x] = 0
                    #continue;

                # computing Y	
                distr = c["distr"]
                out_name_distr[line_count - from_x] = [row["dataset"],distr]

                if (distr == "uniform"):
                    uniform_distr += 1

                if (mode == -1):
                    line_count += 1
                    continue

                fileHist = pathHist + row["dataset"] + "_summary.csv"

                rq0 = rq[row["dataset"] + "-" + row["numQuery"]]

                # computing X in the different cases
                # mode == 0: local emb, global emb, rq emb
                if (mode == 0):
                    hist_RQ = gen_rq_layer(rq0, DIM_H_X, DIM_H_Y)
                    out_hist_rq[line_count-from_x] = hist_RQ
                    embL, embG, embRQ = get_embedding(local_enc, global_enc, hist_RQ, fileHist, mbr[row["dataset"]])
                    embL = embL.numpy().reshape((DIM_E_X, DIM_E_Y, DIM_E_Z))
                    embG = embG.numpy().reshape((32, 32, 2))
                    embRQ = embRQ.numpy().reshape((32, 32, 2))
                    x = np.concatenate([embL, embG, embRQ], axis=2)
                # mode == 1: local emb, global emb, rq hist
                elif (mode == 1):
                    h_zero = np.zeros((DIM_H_X,DIM_H_Y))
                    # la chiamata di get_embedding non cambia ma embRQ viene buttato in questo caso!
                    embL, embG, embRQ = get_embedding(local_enc, global_enc, h_zero , fileHist, mbr[row["dataset"]])
                    embL = embL.numpy().reshape((DIM_E_X, DIM_E_Y, DIM_E_Z))
                    embG = embG.numpy().reshape((32, 32, 2))
                    embRQ = 0
                    x = np.concatenate([embL, embG, hist_RQ], axis=2)
                # mode == 2: local emb and in a separate array global mbr and rq mbr
                elif (mode == 2):
                    h_zero = np.zeros((DIM_H_X,DIM_H_Y))
                    # la chiamata di get_embedding non cambia ma embRQ viene buttato in questo caso!
                    embL, embG, embRQ = get_embedding(local_enc, global_enc, h_zero , fileHist, mbr[row["dataset"]])
                    embL = embL.numpy().reshape((DIM_E_X, DIM_E_Y, DIM_E_Z))
                    embRQ = 0
                    embG = 0
                    x = embL
                    x1 = np.zeros((8))
                    x1[0] = mbr[row["dataset"]]["minx"]
                    x1[1] = mbr[row["dataset"]]["miny"]
                    x1[2] = mbr[row["dataset"]]["maxx"]
                    x1[3] = mbr[row["dataset"]]["maxy"]
                    x1[4] = rq[row["dataset"] + "-" + row["numQuery"]]["minx"]
                    x1[5] = rq[row["dataset"] + "-" + row["numQuery"]]["miny"]
                    x1[6] = rq[row["dataset"] + "-" + row["numQuery"]]["maxx"]
                    x1[7] = rq[row["dataset"] + "-" + row["numQuery"]]["maxy"]
                    out_x1[line_count - from_x] = x1
                else:
                    # in the other cases x and x1 are useless
                    x = 0.0
                    x1 = 0.0

                out_x[line_count - from_x] = x

                if (flag_sel_card == 0):
                    out_y[line_count-from_x] = float(row["cardinality"]) / c["numFeatures"]
                else:
                    if (flag_sel_card == 1):
                        out_y[line_count-from_x] = float(row["cardinality"])
                    elif (flag_sel_card == 2):
                        out_y[line_count-from_x] = float(row["mbrTests"])
                    else:
                        out_y[line_count-from_x] = float(row["mbrTests"]) / c["numFeatures"]

                line_count += 1
                if (line_count % 100 == 0):
                    print("line: ", str(line_count))

                #if (line_count == 10):
                #    out_y = gh.nor_with_min_max(out_y, 100, 0.0, max_selectivity)
                #    return out_x, out_y, out_hist_rq
                #out_y = gh.nor_y_ab(out_y, 200, 0.0, max_selectivity)

            print("Uniform distr: ",str(uniform_distr)+"/"+str(total))

            # filter out the data points that violate constraints
            out_x = out_x[keep==1]
            out_y = out_y[keep==1]            
            out_x1 = out_x1[keep==1]
            out_hist_rq = out_hist_rq[keep==1]
            out_name_distr = out_name_distr[keep==1]
            return out_x, out_y, out_hist_rq, out_x1, out_name_distr

def gen_rq_layer(rq, dimx, dimy):
    rq_layer = np.zeros((dimx, dimy, 1))

    xsizeG = (X_MAX_REF - X_MIN_REF) / dimx
    ysizeG = (Y_MAX_REF - Y_MIN_REF) / dimy

    cell_area = xsizeG * ysizeG

    start_cell_row = math.floor((rq["miny"] - Y_MIN_REF) / ysizeG)
    if (start_cell_row < 0):
        start_cell_row = 0
    if (start_cell_row > (dimy - 1)):
        start_cell_row = dimy
    start_cell_col = math.floor((rq["minx"] - X_MIN_REF) / xsizeG)
    if (start_cell_col < 0):
        start_cell_col = 0
    if (start_cell_col > (dimx - 1)):
        start_cell_col = dimx
    end_cell_row = math.floor((rq["maxy"] - Y_MIN_REF) / ysizeG)
    if (end_cell_row < 0):
        end_cell_row = -1
    if (end_cell_row > (dimy - 1)):
        end_cell_row = (dimy - 1)
    end_cell_col = math.floor((rq["maxx"] - X_MIN_REF) / xsizeG)
    if (end_cell_col < 0):
        end_cell_col = -1
    if (end_cell_col > (dimx - 1)):
        end_cell_col = (dimx - 1)
    for i in range(start_cell_row, end_cell_row + 1):
        for j in range(start_cell_col, end_cell_col + 1):

            cell_x_min = X_MIN_REF + j * xsizeG
            cell_x_max = cell_x_min + xsizeG
            cell_y_min = Y_MIN_REF + i * ysizeG
            cell_y_max = cell_y_min + ysizeG
            rq_layer[i, j] = gh.area_intersection((rq['minx'], rq['miny']), (rq['maxx'], rq['maxy']), (cell_x_min, cell_y_min),
                                                  (cell_x_max, cell_y_max)) / cell_area
    return rq_layer

def analyse_and_balance(from_x, to_x, rqFile, flag_num_query, mbrFile, resultFile, cardFile, pathHist, delim, queryNum_start_from):
    #
    # queryNum_start_from: 0 or 1 according to the number given to the first query
    # Reading RQ file
    rq = {}
    with open(rqFile, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        line_count = 0
        rq_count = queryNum_start_from
        dName_old = ""
        for row1 in csv_reader:
            if (line_count == 0):
                #print(f'Column names are: {", ".join(row1)}')
                dName_old = row1["datasetName"]
            if (line_count < 10):
                print(f'\t{row1["datasetName"]},{row1["numQuery"]}: {row1["minX"]}, {row1["minY"]}, {row1["maxX"]}, {row1["maxY"]}.')
            dName = row1["datasetName"]
            if (dName != dName_old):
                rq_count = queryNum_start_from
            if (flag_num_query == 0):
                name = dName + "-" + str(rq_count)
            else:
                name = dName + "-" + row1["numQuery"]
            print(name)
            rq[name] = dict([('minx', float(row1["minX"])), ('miny', float(row1["minY"])), ('maxx', float(row1["maxX"])),
                             ('maxy', float(row1["maxY"]))])
            rq_count += 1
            dName_old = dName
            line_count += 1
    print("Processed file with query rectangles: ", line_count)
    #
    # Reading MBR file
    mbr = {}
    with open(mbrFile, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file,delimiter=',')
        line_count = 0
        for row in csv_reader:
            if (line_count < 10):
                print(f'\t{row["datasetName"]}: {row["minX"]}, {row["minY"]}, {row["maxX"]}, {row["maxY"]}.')
            name = row["datasetName"]
            mbr[name] = dict([('minx', float(row["minX"])), ('miny', float(row["minY"])), ('maxx', float(row["maxX"])), ('maxy', float(row["maxY"]))])
            line_count += 1
    print("Processed file with dataset MBRs: ", line_count)
    #
    # Reading Cardinality of datasets
    card = {}
    with open(cardFile, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if (line_count < 10):
                print(f'\t{row["datasetName"]}: {row["num_features"]}.')
            name = row["datasetName"]
            card[name] = dict([('numFeatures', float(row["num_features"])), ('distr', row["distribution"])])
            line_count += 1
    print("Processed file with dataset characteristics: ", line_count)
    #
    # Reading Result file
    rq_res = {}
    total = to_x - from_x
    out_y = np.zeros((total))
    out_rq_id = np.empty((total,), dtype='<U60')

    with open(resultFile, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file,delimiter=delim)
        line_count = 0
        count = 0
        uniform_distr = 0
        for row in csv_reader:
            if (line_count == to_x):
                break
            if (line_count < from_x):
                line_count += 1
                continue;

            # Building new record for result file
            # dataset 
            dname = row["dataset"]
            d_prop = card[dname]
            d_mbr = mbr[dname]
            distr = d_prop["distr"]
            d_card = d_prop["numFeatures"]
            # range query
            rq_id = row["dataset"] + "-" + row["numQuery"]
            rq_mbr = rq[rq_id]
            # Computing y
            rq_card = float(row["cardinality"])
            y = rq_card / d_card
            # print("y: ",y)
            rq_res[rq_id] = dict([('dataset', dname),('distr', distr), ('card', d_card), \
                       ('minx', d_mbr["minx"]), ('miny', d_mbr["miny"]), ('maxx', d_mbr["maxx"]), ('maxy', d_mbr["maxy"]), \
                       ('rq_minx', rq_mbr["minx"]), ('rq_miny', rq_mbr["miny"]), ('rq_maxx', rq_mbr["maxx"]), ('rq_maxy', rq_mbr["maxy"]), \
                       ('rq_card', rq_card), ('rq_sel', y)])
            out_y[line_count-from_x] = y
            out_rq_id[line_count-from_x] = rq_id

            if (distr == "uniform"):
                uniform_distr += 1

            line_count += 1
            if (line_count % 100 == 0):
                print("line: ", str(line_count))

    numclass = 10
    freq = gh.count_frequency_1(out_y,numclass-1,1)
    print("Initial distribution")
    p.plot_freq(freq)
    # uniform, parcel, gauss, diagonal, bit, sierpinski, real
    numdistr = 1 # only real

    print("Instances per class (",numclass,") of y and ",numdistr," distributions: ", int(out_y.shape[0]/(numclass*numdistr)))
    freq_to_find = np.full((numclass,numdistr), int(out_y.shape[0]/(numclass*numdistr)))
    keep = np.zeros((total), dtype=np.int8)

    for i in range(out_y.shape[0]):
        index0 = math.ceil(out_y[i]*(numclass-1))
        distr = rq_res[out_rq_id[i]]['distr']
        if (distr == "real"):
            index1 = 0
        elif (distr == "parcel"):
            index1 = 1
        elif (distr == "gaussian"):
            index1 = 2
        elif (distr == "diagonal"):
            index1 = 3
        elif (distr == "bit"):
            index1 = 4
        elif (distr == "sierpinski"):
            index1 = 5
        elif (distr == "uniform"):
            index1 = 6
        if (freq_to_find[index0,index1] > 0):
            keep[i] = 1
            freq_to_find[index0,index1] -= 1

    out_y_kept = out_y[keep==1]
    out_rq_id_kept = out_rq_id[keep==1]

    out_y = out_y[keep==0]
    out_rq_id = out_rq_id[keep==0]

    freq = gh.count_frequency_1(out_y,numclass-1,1)
    print("Remaining...")
    p.plot_freq(freq)

    # ---------------------------------------------
    # FIRST enlarge
    print("First enlarge rq_mbr...")
    out_y, rq_res = enlarge(out_y, rq_res, out_rq_id, pathHist)

    # append new data points
    freq = gh.count_frequency_1(out_y,numclass-1,1)
    p.plot_freq(freq)
    keep = np.zeros((out_y.shape[0]), dtype=np.int8)

    for i in range(out_y.shape[0]):
        index0 = math.ceil(out_y[i]*(numclass-1))
        distr = rq_res[out_rq_id[i]]['distr']
        if (distr == "real"):
            index1 = 0
        elif (distr == "parcel"):
            index1 = 1
        elif (distr == "gaussian"):
            index1 = 2
        elif (distr == "diagonal"):
            index1 = 3
        elif (distr == "bit"):
            index1 = 4
        elif (distr == "sierpinski"):
            index1 = 5
        elif (distr == "uniform"):
            index1 = 6
        if (freq_to_find[index0,index1] > 0):
            keep[i] = 1
            freq_to_find[index0,index1] -= 1

    for j in range(numclass):
        print("class [",j,"][u,p,g,d,b,s,r]: ",freq_to_find[j])

    out_y_kept = np.concatenate((out_y_kept,out_y[keep==1]))
    out_rq_id_kept = np.concatenate((out_rq_id_kept,out_rq_id[keep==1]))

    # ---------------------------------------------
    # OTHER enlarge
    rep = 0
    while True:
        num_dp = out_y_kept.shape[0]
        out_y = out_y[keep==0]
        out_rq_id = out_rq_id[keep==0]

        freq = gh.count_frequency_1(out_y,numclass-1,1)
        print("Remaining ",rep," step...")
        p.plot_freq(freq)
        print(rep," enlarge rq_mbr...")
        out_y, rq_res = enlarge(out_y, rq_res, out_rq_id, pathHist)

        # append new data points
        freq = gh.count_frequency_1(out_y,numclass-1,1)
        p.plot_freq(freq)
        keep = np.zeros((out_y.shape[0]), dtype=np.int8)

        for i in range(out_y.shape[0]):
            index0 = math.ceil(out_y[i]*(numclass-1))
            distr = rq_res[out_rq_id[i]]['distr']
            if (distr == "real"):
                index1 = 0
            elif (distr == "parcel"):
                index1 = 1
            elif (distr == "gaussian"):
                index1 = 2
            elif (distr == "diagonal"):
                index1 = 3
            elif (distr == "bit"):
                index1 = 4
            elif (distr == "sierpinski"):
                index1 = 5
            elif (distr == "uniform"):
                index1 = 6
            if (freq_to_find[index0,index1] > 0):
                keep[i] = 1
                freq_to_find[index0,index1] -= 1

        for j in range(numclass):
            print("class [",j,"][r,p,g,d,b,s,u]: ",freq_to_find[j])

        out_y_kept = np.concatenate((out_y_kept,out_y[keep==1]))
        out_rq_id_kept = np.concatenate((out_rq_id_kept,out_rq_id[keep==1]))

        if (num_dp == out_y_kept.shape[0] or rep == 10):
            break
        rep += 1

    print("Final state - num data points: ", out_y_kept.shape[0])
    freq = gh.count_frequency_1(out_y_kept,numclass-1,1)
    p.plot_freq(freq)

    np.save("y_"+str(out_y_kept.shape[0])+"_rq_balanced", out_y_kept)
    # save of new rq_resul_file TODO
    fieldnames = ['dataset', 'distr', 'card', 'minx', 'miny', 'maxx', 'maxy', 'rq_minx', 'rq_miny', 'rq_maxx', 'rq_maxy', 'rq_card', 'rq_sel']
    with open("rq_result_"+str(out_y_kept.shape[0])+"_balanced.csv", 'w', encoding='UTF8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(out_y_kept.shape[0]):
            writer.writerow(rq_res[out_rq_id_kept[i]])

def enlarge (out_y, rq_res, out_rq_id, pathHist):
    print("Enlarge rq_mbr...")
    for i in range(out_y.shape[0]):
        if (i % 100 == 0):
            print("Done: ", i)
        rec = rq_res[out_rq_id[i]]
        deltax = rec["rq_maxx"] - rec["rq_minx"]
        deltay = rec["rq_maxy"] - rec["rq_miny"]

        dd = rd.random()
        if (dd < 0.3):
            if (deltax*100000 < 10.0):
                deltax = deltax*100000
            if (deltay*100000 < 10.0):
                deltay = deltay*100000

            if (deltax*10000 < 10.0):
                deltax = deltax*10000
            if (deltay*10000 < 10.0):
                deltay = deltay*10000

            if (deltax*1000 < 10.0):
                deltax = deltax*1000
            if (deltay*1000 < 10.0):
                deltay = deltay*1000

            if (deltax*100 < 10.0):
                deltax = deltax*100
            if (deltay*100 < 10.0):
                deltay = deltay*100

            if (deltax*10 < 10.0):
                deltax = deltax*10
            if (deltay*10 < 10.0):
                deltay = deltay*10 
        else:
            if (deltax*5 < 10.0):
                deltax = deltax*5
            if (deltay*5 < 10.0):
                deltay = deltay*5

        new_rq = dict([('minx', float(rec["rq_minx"]-(deltax/2))), ('miny', float(rec["rq_miny"]-(deltay/2))), \
                       ('maxx', float(rec["rq_maxx"]+(deltax/2))), ('maxy', float(rec["rq_maxy"]+(deltay/2)))])
        fileHist = pathHist + rec["dataset"] + "_summary.csv"
        hist_local = gh.gen_hist_from_file(DIM_H_X,DIM_H_Y,DIM_H_Z,fileHist)
        global X_MAX_REF
        global X_MIN_REF
        global Y_MAX_REF
        global Y_MIN_REF
        X_MIN_REF = rec['minx']
        X_MAX_REF = rec['maxx']
        Y_MIN_REF = rec['miny']
        Y_MAX_REF = rec['maxy']

        hist_RQ = gen_rq_layer(new_rq, DIM_H_X, DIM_H_Y)
        new_rq_card = 0
        for j in range(DIM_H_X):
            for k in range(DIM_H_Y):
                new_rq_card += hist_RQ[j,k,0] * hist_local[j,k,0]
        y = new_rq_card / rec['card']
        if (y > 1.0):
            y = 1.0
        #print("Old card: ", rec['rq_card'], " New card: ", new_rq_card)
        #print("Old rq: ", rec['rq_minx'], rec['rq_miny'], rec['rq_maxx'], rec['rq_maxy'], "New rq: ", new_rq["minx"], new_rq["miny"], new_rq["maxx"], new_rq["maxy"])
        rq_res[out_rq_id[i]] = dict([('dataset', rec['dataset']),('distr', rec['distr']), ('card', rec['card']), \
                       ('minx', rec["minx"]), ('miny', rec["miny"]), ('maxx', rec["maxx"]), ('maxy', rec["maxy"]), \
                       ('rq_minx', new_rq["minx"]), ('rq_miny', new_rq["miny"]), ('rq_maxx', new_rq["maxx"]), ('rq_maxy', new_rq["maxy"]), \
                       ('rq_card', new_rq_card), ('rq_sel', y)])
        out_y[i] = y
    return out_y, rq_res
