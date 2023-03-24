#!/usr/bin/env python3
import os
from os import listdir, mkdir, path, sep
from os.path import isfile, join
import numpy as np
import random as rd
import math
import csv
import generate_histogram as gh
from tensorflow import keras
import plot as p
import datetime as dt
import logging

# CONST -------------------------
#
DIM_H_X = 128
DIM_H_Y = 128
DIM_H_Z = 6

DIM_HG_Z = 1

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
X_MAX_REF = 20
Y_MIN_REF = 0
Y_MAX_REF = 20

# ----------------------------------------
# rewriting of the code for the join case
# ----------------------------------------

def get_embedding(local_enc, global_enc, datasetFile, mbr):
#
# PARAMETERS:
# local_enc: model for generating the local embeddings
# global_enc: model for generating the global embeddings
# datasetFile: name of the file containing the local histogram
# mbr: MBR of the dataset
#
	# get local histogram -----------------------------
	hist_local = gh.gen_hist_from_file(DIM_H_X,DIM_H_Y,DIM_H_Z,datasetFile)
	#print("hist_local: ", hist_local.shape)
	hist_local_norm, min_a, max_a = gh.nor_g_ab(hist_local.reshape((1,DIM_H_X,DIM_H_Y,DIM_H_Z)), 1, NORM_MIN, NORM_MAX)
	#print("hist_local_norm: ", hist_local_norm.shape)
	emb_local = local_enc.encoder(hist_local_norm.reshape((1,DIM_H_X,DIM_H_Y,DIM_H_Z)))

	# computing global histogram ----------------------
	hist_glob = gh.gen_global_hist(hist_local, DIM_H_X, DIM_H_Y, mbr)
	hist_glob_norm, min_g, max_g  = gh.nor_g_ab(hist_glob.reshape((1,DIM_H_X,DIM_H_Y)), 1, NORM_MIN_G, NORM_MAX_G)
	emb_global = global_enc.encoder(hist_glob_norm.reshape((1,DIM_H_X,DIM_H_Y,DIM_HG_Z)))

	return emb_local, emb_global


def gen_inputs_embs_jn (mode, flag_sel_card, from_x, to_x, file_result, file_summary, path_hist, delim, max_y, emb, embt, dt):
# flag_sel_card: 0 stores in y selectivity,
#                1 cardinality,
#                2 mbrTests,
#                3 mbrTests selecivity
# mode: 0 the LOCAL and GLOBAL embeddings of both datasets are computed and added to x and x1 respectively,,
#       1 the LOCAL embeddings of both datasets are computed and added to x,
#         and the vector containing the MBR of both datasets are added to xg
#       2 only y is generated
#       file_result: file containing the results of the queries 'join/join_res_small.csv'
#       file_summary: file containing the characteristics of the datasets 'dataset-summaries.csv'
#       path_hist: path where are stored the histograms 'histograms/new_datasets/'
#       max_y: max acceptable values for y; values of y greater than 1.1*max_y are discarded
#	emb: autoencoder to be used for generating the embeddings; 0=emb0, 1=emb1, ...
#	embt: type of autoencoder: trained with 'synt' data or trained with synt+'real' data  
#	dt: 'synt' or 'real' or 'real_er' excluding rotation

	# LOCAL embeddings
	# decomment the models you want to use for embeddings generation,
	# change also the file name for preserving previous generated files
	# and the DIM_E_X, ...

	if (embt == 'synt'):
		# embedding DENSE synthetic only
		f_emb0 = 'model/autoencoder_3072_CNNDense_newDatasets_SMLG';
		f_emb1 = 'model/autoencoder_DENSE3L_1024-512_emb384_synthetic';
		f_emb2 = 'model/autoencoder_DENSE3L_1024-512_emb1536_synthetic';
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

	if (embt == 'synt'):
		print("autoencoder trained with synthetic data")
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
		print("autoencoder trained with synthetic+real data")
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

	# num = to_x - from_x

	out_x, out_x1, out_ds, out_y = gen_join_input_from_file(mode, flag_sel_card, from_x, to_x, e, g0, file_summary, file_result, path_hist, delim, max_y, dt)
	np.save("x_"+str(out_x.shape[0])+"_jn_"+str(mode)+"_"+emb_txt, out_x)
	
	if (mode == 2):
		np.save("y_"+str(out_y.shape[0])+"_jn_"+str(flag_sel_card),out_y)
		return
	else:
		print("skip")
		np.save("x1_"+str(out_x1.shape[0])+"_jn_"+str(mode),out_x1)
		np.save("ds_"+str(out_ds.shape[0])+"_jn",out_ds)
		np.save("y_"+str(out_y.shape[0])+"_jn_"+str(flag_sel_card),out_y)

	
def gen_join_input_from_file(mode, flag_sel_card, from_x, to_x, local_enc, global_enc, mbrFile, resultFile, pathHist, delim, max_y, dt):
#
# PARAMETERS:
# flag_sel_card: 0 stores in y selectivity, 
#                1 cardinality, 
#                2 mbrTests,
#                3 mbrTests selectivity
# mode: 0 the LOCAL and GLOBAL embeddings of both datasets are computed and added to x and xg respectively, 
#       1 the LOCAL embeddings of both datasets are computed and added to x, 
#         and the vector containing the MBR of both datasets are added to xg
# local_enc: encoder for local embedding
# global_enc: encorder for global embedding
# mbrFile: name of the file containing the mbr of the datasets (datasetSummary)
# resultFile: name of the file containing the results of the join operations
# pathHist: directory where the histograms are stored
# delim: delimiter of the resultFile
# max_y: max acceptable values for y; values of y greater than 1.1*max_y are discarded
# dt: 'synt' or 'real' or 'real_er' excluding rotation
# 
# --------------------------------------------------------------------------
# Reading SUMMARY file and extracting for each dataset: MBR, features and distribution
#
	print("Reading the datasets summary file...")
	mbr = {}
	features = {}
	distr = {}
	with open(mbrFile, mode='r') as csv_file:
		csv_reader = csv.DictReader(csv_file,delimiter=delim)
		line_count = 0
		for row in csv_reader:
			if (line_count == 0):
				print(f'Column names are: {", ".join(row)}')
			if ((dt == "real") or (dt == "real_er")):
				print(f'\t{row["datasetName"]},{row["distribution"]}: {row["minX"]}, {row["minY"]}, {row["maxX"]}, {row["maxY"]}.')
			else:
				print(f'\t{row["datasetName"]},{row["distribution"]}: {row["x1"]}, {row["y1"]}, {row["x2"]}, {row["y2"]}.')
				
			name = row["datasetName"]
			if ((dt == "real") or (dt == "real_er")):
				mbr[name] = dict([('minx', float(row["minX"])), ('miny', float(row["minY"])), ('maxx', float(row["maxX"])), ('maxy', float(row["maxY"]))])
			else:
				mbr[name] = dict([('minx', float(row["x1"])), ('miny', float(row["y1"])), ('maxx', float(row["x2"])), ('maxy', float(row["y2"]))])
			distr[name] = dict([('group', row["distribution"])])
			features[name] = dict([('card', float(row["num_features"])),('size', float(row["size"])),('numPnts', float(row["num_points"])), ('avgArea', float(row["avg_area"])), ('avgLenX', float(row["avg_side_length_0"])), ('avgLenY', float(row["avg_side_length_1"])), ('E0', -float(row["E0"])), ('E2', -float(row["E2"]))  ])   
			line_count += 1
# --------------------------------------------------------------------------
# Reading Result file
#
	print("Reading the file with join results...")
	total = to_x - from_x
	out_y = np.zeros((total))
	out_x = np.zeros((total,DIM_E_X,DIM_E_Y,(2*DIM_E_Z)))
	out_distr = np.zeros((total,2))
	keep = np.ones((total), dtype=np.int8)

	if (mode == 0):
		out_xg = np.zeros((total,DIM_E_X,DIM_E_Y,(2*DIM_EG_Z)))
	else:
		out_xg = np.zeros((total, 8))

	with open(resultFile, mode='r') as csv_file:
		csv_reader = csv.DictReader(csv_file,delimiter=',')
		line_count = 0
		count = 0
		uniform_distr = 0
		for row in csv_reader:
			if (line_count == 0):
				print("Row number: "+str(line_count))
			if (line_count == to_x):
				break
			if (line_count < from_x):
				line_count += 1
				continue;
                
			file1 = row["dataset1"]
			print("File1: ",file1)
			idx_rot = file1.find("_r")
			if ((dt == "synt") and (idx_rot < 0)):
				idx = file1.find("dataset-")
				idx_pnt = file1.find(".")
				file1 = file1[idx:idx_pnt]
				fileHist1 = pathHist + file1 + "_summary.csv"
				print("File1 modified: ",file1)
			else:
				if ((dt == "real") or (idx_rot < 0)): 
					fileHist1 = pathHist + file1 + "_summary.csv"
					file1 = "lakes_parks/"+file1
				elif (dt == "real_er"):
					fileHist1 = pathHist + file1[0:idx_rot] + ".csv_summary.csv"
					file1 = "lakes_parks/"+file1[0:idx_rot] + ".csv" 

			file2 = row["dataset2"]
			print("File2: ",file2)
			idx_rot = file2.find("_r")
			if ((dt == "synt") and (idx_rot < 0)):
				idx = file2.find("dataset-")
				idx_pnt = file2.find(".")
				file2 = file2[idx:idx_pnt]
				fileHist2 = pathHist + file2 + "_summary.csv"
				print("File2 modified: ",file2)
			else:
				if ((dt == "real") or (idx_rot < 0)):
					fileHist2 = pathHist + file2 + "_summary.csv"
					file2 = "lakes_parks/"+file2
				elif (dt == "real_er"):
					fileHist2 = pathHist + file2[0:idx_rot] + ".csv_summary.csv"
					file2 = "lakes_parks/"+file2[0:idx_rot] +".csv"

			#if not (file1 in mbr.keys()):
			#	print ("mbr not found for key: ",file1)
			#	continue 
			#if not (file2 in mbr.keys()):
			#	print ("mbr not found for key: ",file2)
			#	continue

		 	# computing X in the different cases
			# mode == 0: local emb, global emb
			if (mode == 0):
				embL1, embG1 = get_embedding(local_enc, global_enc, fileHist1, mbr[file1])
				embL1 = embL1.numpy().reshape((DIM_E_X, DIM_E_Y, DIM_E_Z))
				embG1 = embG1.numpy().reshape((32, 32, 2))
				embL2, embG2 = get_embedding(local_enc, global_enc, fileHist2, mbr[file2])
				embL2 = embL2.numpy().reshape((DIM_E_X, DIM_E_Y, DIM_E_Z))
				embG2 = embG2.numpy().reshape((32, 32, 2))
				x = np.concatenate([embL1, embL2], axis=2)
				xg = np.concatenate([embG1, embG2], axis=2)
			# mode == 1: local emb and in a separate array mbr of both datasets
			else:
				embL1, embG1 = get_embedding(local_enc, global_enc, fileHist1, mbr[file1])
				embL1 = embL1.numpy().reshape((DIM_E_X, DIM_E_Y, DIM_E_Z))
				embG1 = embG1.numpy().reshape((32, 32, 2))
				embL2, embG2 = get_embedding(local_enc, global_enc, fileHist2, mbr[file2])
				embL2 = embL2.numpy().reshape((DIM_E_X, DIM_E_Y, DIM_E_Z))
				embG2 = embG2.numpy().reshape((32, 32, 2))
				x = np.concatenate([embL1, embL2], axis=2)
				xg = np.zeros((8))
				
				xg[0] = mbr[file1]["minx"]
				xg[1] = mbr[file1]["miny"]
				xg[2] = mbr[file1]["maxx"]
				xg[3] = mbr[file1]["maxy"]
				xg[4] = mbr[file2]["minx"]
				xg[5] = mbr[file2]["miny"]
				xg[6] = mbr[file2]["maxx"]
				xg[7] = mbr[file2]["maxy"]

			# computing Y in different cases
			# flag_sel_card == 0: selectivity
			if (flag_sel_card == 0):
				c1 = features[file1]["card"]
				c2 = features[file2]["card"]
				y = float(row["resultSJSize"]) / (c1*c2)
				if (y >= 1.1*max_y):
					print("continue: ",str(line_count - from_x))
					keep[line_count - from_x] = 0
			elif (flag_sel_card == 1):
				y = float(row["resultSJSize"])
			elif (flag_sel_card == 2):
				y = float(row["PBSMMBRTests"])
			else:
				c1 = features[file1]["card"]
				c2 = features[file2]["card"]
				y = float(row["PBSMMBRTests"])/(c1*c2)
				print("MBR sel: ", y, "=",float(row["PBSMMBRTests"]), "/", (c1*c2))
				if (y >= 1.1*max_y):
					print("continue: ",str(line_count - from_x))
					keep[line_count - from_x] = 0

			out_x[line_count - from_x] = x
			out_xg[line_count - from_x] = xg
			out_y[line_count-from_x] = y
		
			# storing distribution of datasets
			if (distr[file1]["group"].lower() == "uniform"):
				d1 = 0
			elif (distr[file1]["group"].lower() == "parcel"):
				d1 = 1
			elif (distr[file1]["group"].lower() == "gaussian"):
				d1 = 2
			elif (distr[file1]["group"].lower() == "bit"):
				d1 = 3
			elif (distr[file1]["group"].lower() == "diagonal"):
				d1 = 4
			elif (distr[file1]["group"].lower() == "sierpinski"):
				d1 = 5
			else:
				d1 = 6

			if (distr[file2]["group"].lower() == "uniform"):
				d2 = 0
			elif (distr[file2]["group"].lower() == "parcel"):
				d2 = 1
			elif (distr[file2]["group"].lower() == "gaussian"):		
				d2 = 2
			elif (distr[file2]["group"].lower() == "bit"):
				d2 = 3
			elif (distr[file2]["group"].lower() == "diagonal"):
				d2 = 4
			elif (distr[file2]["group"].lower() == "sierpinski"):
				d2 = 5
			else:
				d2 = 6

			out_distr[line_count-from_x][0] = d1
			out_distr[line_count-from_x][1] = d2

			line_count += 1
			print("line: ", str(line_count))
	out_x = out_x[keep==1]
	out_xg = out_xg[keep==1]
	out_distr = out_distr[keep==1]
	out_y = out_y[keep==1]
	return out_x, out_xg, out_distr, out_y

# -------------------------------------
# Procedure for balancing join results
# -------------------------------------
def analyse_and_balance(c, from_x, to_x, mbrFile, resultFile, pathHist, delim):

# Ideas: 
# 1. fix a maximum for join selectivity, we start with a value C like 0.2 or 0.1
# 2. check the available data to see if they are balanced in the interval: [0..C]
# 3. double the average length on X and Y axis of the geometries of both datasets and, starting from their histograms,
#    compute the new cardinality by applying the multiplication of histograms, i.e. 
#    for each pair of cells ci,cj that intersect themselves compute:
#    num_geo(ci) * num_geo(cj) * (x_i+x_j)*(y_i+y_j)/area(cell)

# --------------------------------------------------------------------------
# Setting logger
	ffh = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
	logger = logging.getLogger()
	logger.setLevel(logging.DEBUG)
	#stdout (console): info level
	ch = logging.StreamHandler()
	ch.setLevel(logging.INFO)
	logger.addHandler(ch)
	# log file: debug level
	fh = logging.FileHandler('myLog.log',mode='w')
	fh.setLevel(logging.DEBUG)
	fh.setFormatter(ffh)
	logger.addHandler(fh)
# --------------------------------------------------------------------------
# Reading SUMMARY file and extracting for each dataset: MBR, features and distribution
#
	logger.info("Reading the datasets summary file...")
	mbr = {}
	features = {}
	distr = {}
	with open(mbrFile, mode='r') as csv_file:
		csv_reader = csv.DictReader(csv_file,delimiter=delim)
		line_count = 0
		for row in csv_reader:
			if (line_count == 0):
				logger.info(f'Column names are: {", ".join(row)}')
			# print(f'\t{row["datasetName"]},{row["distribution"]}: {row["x1"]}, {row["y1"]}, {row["x2"]}, {row["y2"]}.')
			# name = row["datasetName"]
			name = row["dataset"]
			mbr[name] = dict([('minx', float(row["x1"])), ('miny', float(row["y1"])), ('maxx', float(row["x2"])), ('maxy', float(row["y2"]))])
			distr[name] = row["distribution"]
			features[name] = dict([('card', float(row["num_features"])),('size', float(row["size"])),('numPnts', float(row["num_points"])), ('avgArea', float(row["avg_area"])), ('avgLenX', float(row["avg_side_length_0"])), ('avgLenY', float(row["avg_side_length_1"])), ('E0', -float(row["E0"])), ('E2', float(row["E2"]))  ])
			line_count += 1
	logger.info("Processed file with dataset characteristics: "+str(line_count))
# --------------------------------------------------------------------------    
# Reading Result file
	jn_res = {}
	total = to_x - from_x
	out_y = np.zeros((total))
	out_jn_id = np.empty((total,), dtype='<U60')

	logger.info('Reading result file...')
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
			ds1 = row["dataset1"]
			idx = ds1.find("dataset-")
			idx_pnt = ds1.find(".")
			ds1 = ds1[idx:idx_pnt]
			
			distr1 = distr[ds1]
			feature1 = features[ds1]
			mbr1 = mbr[ds1]
			
			ds2 = row["dataset2"]
			idx = ds2.find("dataset-")
			idx_pnt = ds2.find(".")
			ds2 = ds2[idx:idx_pnt]

			distr2 = distr[ds2]
			feature2 = features[ds2]
			mbr2 = mbr[ds2]
			logger.debug("D1: "+ds1+" D2: "+ds2)
			c1 = features[ds1]["card"]
			c2 = features[ds2]["card"]
			jn_card = float(row["resultSJSize"])
			y = jn_card / (c1*c2)
			jn_id = ds1 + "-" + ds2
			jn_res[jn_id] = dict([('ds1', ds1),('distr1', distr1),('ds2', ds2),('distr2', distr2), \
			('card1', feature1["card"]),('card2', feature2["card"]), \
			('avgArea1', feature1["avgArea"]),('avgArea2', feature2["avgArea"]), \
			('avgLenX1', feature1["avgLenX"]),('avgLenX2', feature2["avgLenX"]), \
			('avgLenY1', feature1["avgLenY"]),('avgLenY2', feature2["avgLenY"]), \
			('minx1', mbr1["minx"]), ('miny1', mbr1["miny"]), ('maxx1', mbr1["maxx"]), ('maxy1', mbr1["maxy"]), \
			('minx2', mbr2["minx"]), ('miny2', mbr2["miny"]), ('maxx2', mbr2["maxx"]), ('maxy2', mbr2["maxy"]), \
			('jn_card', jn_card), ('jn_sel', y)])
			out_y[line_count-from_x] = y
			out_jn_id[line_count-from_x] = jn_id

			line_count += 1
			if (line_count % 100 == 0):
				logger.info("line: "+str(line_count))
	
	numclass = 10
	max_y = np.amax(out_y)
	logger.info("MAX JN_SEL (from data): "+str(max_y))
	freq0 = gh.count_frequency_advanced(out_y,numclass,c)
	logger.debug("Initial distribution (0,"+str(c)+")")
	p.plot_freq_adv_log(freq0,10,c,logger)
	# the type of distribution of the datasets is not considered
	logger.info("Instances of y per class("+str(2*numclass+1)+"): "+str(int(out_y.shape[0]/(2*numclass+1))))
	
	freq_to_find = np.full((2*numclass+1), int(out_y.shape[0]/(2*numclass+1)))
	keep = np.zeros((total), dtype=np.int8)

	for i in range(out_y.shape[0]):
		if (out_y[i] == 0):
			index0 = 0
		elif (out_y[i] < c/10):
			index0 = math.ceil(out_y[i]/(c/10)*numclass)
		elif (out_y[i] < c):
			index0 = math.ceil(out_y[i]/c*numclass)+numclass
		else:
			index0 = 2*numclass
		if (freq_to_find[index0] > 0):
			keep[i] = 1
			freq_to_find[index0] -= 1

	out_y_kept = out_y[keep==1]
	out_jn_id_kept = out_jn_id[keep==1]
	
	out_y = out_y[keep==0]
	out_jn_id = out_jn_id[keep==0]

	freq = gh.count_frequency_advanced(out_y,numclass,c)
	logger.debug("Remaining...")
	p.plot_freq_adv_log(freq,numclass,c,logger)

	# ---------------------------------------------
	# FIRST enlargement of geometries area
	factor = 0.8
	logger.info("First enlarge... step: 4, factor: "+str(factor))
	logger.info("In out_y: "+str(out_y.shape[0])+" jn_res: "+str(len(jn_res))+" out_jn_id: "+str(out_jn_id.shape[0]))
	out_y, jn_res, out_jn_id = enlarge(4, factor, out_y, jn_res, out_jn_id, pathHist, logger)
	logger.info("Out out_y: "+str(out_y.shape[0])+" jn_res: "+str(len(jn_res))+" out_jn_id: "+str(out_jn_id.shape[0]))

	logger.info("MAX JN_SEL (from data): "+str(max_y))
	max_y_1 = np.amax(out_y)
	logger.info("MAX JN_SEL (after first enlargement): "+str(max_y_1))
	
	# plot new out_y
	("out_y after the enlargement:")
	freq = gh.count_frequency_advanced(out_y,numclass,c)
	p.plot_freq_adv_log(freq,numclass,c,logger)
	
	# append new data points
	keep = np.zeros((out_y.shape[0]), dtype=np.int8)
	for i in range(out_y.shape[0]):
		if (out_y[i] == 0):
			index0 = 0
		elif (out_y[i] < c/10):
			index0 = math.ceil(out_y[i]/(c/10)*numclass)
		elif (out_y[i] < c):
			index0 = math.ceil(out_y[i]/c*numclass)+numclass
		else:
			index0 = 2*numclass
		if (freq_to_find[index0] > 0):
			keep[i] = 1
			freq_to_find[index0] -= 1

	logger.debug("Missing elements in the classes:")
	for j in range(2*numclass):
		logger.debug("class ["+str(j)+"]: "+str(freq_to_find[j]))
	out_y_kept = np.concatenate((out_y_kept,out_y[keep==1]))
	out_jn_id_kept = np.concatenate((out_jn_id_kept,out_jn_id[keep==1]))

	out_y = out_y[keep==0]
	out_jn_id = out_jn_id[keep==0]

	freq = gh.count_frequency_advanced(out_y,numclass,c)
	logger.debug("Remaining...")
	p.plot_freq_adv_log(freq,numclass,c,logger)

	# ---------------------------------------------
        # SECOND enlargement of geometries area
	logger.info("Second enlarge... step: 8, factor: "+str(factor))
	logger.info("In out_y: "+str(out_y.shape[0])+" jn_res: "+str(len(jn_res))+" out_jn_id: "+str(out_jn_id.shape[0]))
	out_y, jn_res, out_jn_id = enlarge(8, factor, out_y, jn_res, out_jn_id, pathHist, logger)
	logger.info("Out out_y: "+str(out_y.shape[0])+" jn_res: "+str(len(jn_res))+" out_jn_id: "+str(out_jn_id.shape[0]))

	# plot new out_y
	logger.debug("out_y after the second enlargement:")
	freq = gh.count_frequency_advanced(out_y,numclass,c)
	p.plot_freq_adv_log(freq,10,c,logger)
	logger.info("MAX JN_SEL (from data): "+str(max_y))
	max_y_1 = np.amax(out_y)
	logger.info("MAX JN_SEL (after second enlargement): "+str(max_y_1))

	# append new data points
	keep = np.zeros((out_y.shape[0]), dtype=np.int8)
	for i in range(out_y.shape[0]):
		if (out_y[i] == 0):
			index0 = 0
		elif (out_y[i] < c/10):
			index0 = math.ceil(out_y[i]/(c/10)*numclass)
		elif (out_y[i] < c):
			index0 = math.ceil(out_y[i]/c*numclass)+numclass
		else:
			index0 = 2*numclass
		if (freq_to_find[index0] > 0):
			keep[i] = 1
			freq_to_find[index0] -= 1

	logger.debug("Missing elements in the classes:")
	for j in range(2*numclass):
		logger.debug("class ["+str(j)+"]: "+str(freq_to_find[j]))
	out_y_kept = np.concatenate((out_y_kept,out_y[keep==1]))
	out_jn_id_kept = np.concatenate((out_jn_id_kept,out_jn_id[keep==1]))

	logger.info("Final state - num data points: "+str(out_y_kept.shape[0]))
	freqN = gh.count_frequency_advanced(out_y_kept,numclass,c)
	p.plot_freq_adv_log(freqN,numclass,c,logger)

	np.save("y_"+str(out_y_kept.shape[0])+"_jn_balanced", out_y_kept)
	# save of new jn_resul_file
	fieldnames = ['ds1', 'distr1', 'ds2', 'distr2', 'card1', 'card2', \
			'avgArea1', 'avgArea2', 'avgLenX1', 'avgLenX2', 'avgLenY1', 'avgLenY2', 'minx1', 'miny1', 'maxx1', 'maxy1', \
			'minx2', 'miny2', 'maxx2', 'maxy2', 'jn_card', 'jn_sel'] 

	with open("jn_result_"+str(out_y_kept.shape[0])+"_balanced.csv", 'w', encoding='UTF8', newline='') as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		for i in range(out_y_kept.shape[0]):
			if (out_jn_id_kept[i] != ''):
				logger.debug("Saved: "+out_jn_id_kept[i])
				writer.writerow(jn_res[out_jn_id_kept[i]])
			else:
				logger.warning("Skip case: "+str(i))	

def enlarge (step, factor, out_y, jn_res, out_jn_id, pathHist, logger):
# factor: 0.01 to 1. It is the factor of enlargement to be applied to the dimension of the cell of the histogram. 
# If deltaX is the side of cell on X axis, lenX is enlarged as follows: lenX += factor deltaX. The same for Y.
#
	logger.info("Enlarge geometries...")
	logger.info("Enlargement rate: "+str(step))
	now_0 = dt.datetime.now()
	for i in range(out_y.shape[0]):
		now = dt.datetime.now()
		#if (i % 1000 == 0):
		if (i > 0):
			logger.info("Done: "+str(i))
		logger.debug("-----------------------------------")
		rec = jn_res[out_jn_id[i]]
		# D1
		card1 = rec["card1"]
		deltax1 = (rec["maxx1"] - rec["minx1"])/(DIM_H_X)
		deltay1 = (rec["maxy1"] - rec["miny1"])/(DIM_H_Y)
		area1 = rec["avgArea1"]
		lenX1 = rec["avgLenX1"]
		lenY1 = rec["avgLenY1"]
		# D2
		card2 = rec["card2"]
		deltax2 = (rec["maxx2"] - rec["minx2"])/(DIM_H_X)
		deltay2 = (rec["maxy2"] - rec["miny2"])/(DIM_H_Y)
		area2 = rec["avgArea2"]
		lenX2 = rec["avgLenX2"]
		lenY2 = rec["avgLenY2"]

		logger.debug("D1: "+rec["ds1"]+" "+rec["distr1"]+" lenX,lenY: "+str(lenX1)+" "+str(lenY1)+" MBR: "+str(rec["minx1"])+" "+str(rec["maxx1"])+" "+str(rec["miny1"])+" "+str(rec["maxy1"])) 
		logger.debug("Cell 1: "+str(deltax1)+" "+str(deltay1))
		logger.debug("D2: "+rec["ds2"]+" "+rec["distr2"]+" lenX,lenY: "+str(lenX2)+" "+str(lenY2)+" MBR: "+str(rec["minx2"])+" "+str(rec["maxx2"])+" "+str(rec["miny2"])+" "+str(rec["maxy2"])) 
		logger.debug("Cell 2: "+str(deltax2)+" "+str(deltay2))

		if (((lenX1 >= deltax1*step) or (lenY1 >= deltay1*step)) and ((lenX2 >= deltax2*step) or (lenY2 >= deltay2*step))):
			# out_y[i] = as before
			#jn_res[out_jn_id[i]] = dict([('ds1', rec['ds1']),('distr1', rec['distr1']),('ds2', rec['ds2']),('distr2', rec['distr2']), \
			#('card1', card1),('card2', card2), \
			#('avgLenX1', lenX1),('avgLenX2', lenX2),('avgLenY1', lenY1),('avgLenY2', lenY2), \
			#('avgArea1', area1 ),('avgArea2', area2 ), \
			#('minx1', rec["minx1"]), ('miny1', rec["miny1"]), ('maxx1', rec["maxx1"]), ('maxy1', rec["maxy1"]), \
			#('minx2', rec["minx2"]), ('miny2', rec["miny2"]), ('maxx2', rec["maxx2"]), ('maxy2', rec["maxy2"]), \
			#('jn_card', rec["jn_card"]), ('jn_sel', out_y[i])])
			logger.debug("Geometries too big, enlargement not applied, skip: "+rec['ds1']+"-"+rec['ds2'])
			continue;
		name1 = rec["ds1"]
		name2 = rec["ds2"]
		# to drop when the save of the new histograms is implemented
		#if (step > 1 and rec["ds1"].find("_e") > 0):
		#	idx = rec["ds1"].find("_e")
		#	name1 = rec["ds1"][0:idx]
		#if (step > 1 and rec["ds2"].find("_e") > 0):
		#	idx = rec["ds2"].find("_e")
		#	name2 = rec["ds2"][0:idx]
		fileHist1 = pathHist + name1 + "_summary.csv"
		fileHist2 = pathHist + name2 + "_summary.csv"

		hist_1 = gh.gen_hist_from_file(DIM_H_X,DIM_H_Y,DIM_H_Z,fileHist1)
		hist_2 = gh.gen_hist_from_file(DIM_H_X,DIM_H_Y,DIM_H_Z,fileHist2)
		hist_1_new = np.zeros((DIM_H_X,DIM_H_Y,DIM_H_Z)) 
		hist_2_new = np.zeros((DIM_H_X,DIM_H_Y,DIM_H_Z)) 
		
		#print("hist_1.shape: ", hist_1.shape)
		#gh.prr_orig(hist_1[:,:,0])
		#print("hist_2.shape: ", hist_2.shape)
		#gh.prr_orig(hist_2[:,:,0])
		
		for j1 in range(DIM_H_X):
			for k1 in range(DIM_H_Y):
				new_lenX1 = hist_1[j1,k1,4] + deltax1*step*factor
				new_lenY1 = hist_1[j1,k1,5] + deltay1*step*factor
				new_area1 = new_lenX1 * new_lenY1
				new_lenX2 = hist_2[j1,k1,4] + deltax2*step*factor
				new_lenY2 = hist_2[j1,k1,5] + deltay2*step*factor
				new_area2 = new_lenX2 * new_lenY2
				# update histograms
				hist_1_new[j1,k1,0] = hist_1[j1,k1,0]
				hist_1_new[j1,k1,1] = hist_1[j1,k1,1]
				hist_1_new[j1,k1,2] = hist_1[j1,k1,2]
				hist_1_new[j1,k1,3] = new_area1 
				hist_1_new[j1,k1,4] = new_lenX1
				hist_1_new[j1,k1,5] = new_lenY1
				# update histograms
				hist_2_new[j1,k1,0] = hist_2[j1,k1,0]
				hist_2_new[j1,k1,1] = hist_2[j1,k1,1]
				hist_2_new[j1,k1,2] = hist_2[j1,k1,2]
				hist_2_new[j1,k1,3] = new_area2
				hist_2_new[j1,k1,4] = new_lenX2
				hist_2_new[j1,k1,5] = new_lenY2


		logger.debug("Computing histograms multiplication...")
		new_jn_card = 0.0
		n_prob_uno = 0
		n_prob_tot = 0
		#int_cell = np.zeros((DIM_H_X,DIM_H_Y))
		for j1 in range(DIM_H_X):
			for k1 in range(DIM_H_Y):
				# notice that: the spatial location of the cell (0,0) is in the bottom left corner, moreover
				# j1 is generating the y coordinate and k1 is generating the x coordinate
				cell1_x = rec["minx1"]+deltax1*k1 
				cell1_y = rec["miny1"]+deltay1*j1
				area_ov_mbr = intersection(cell1_x, cell1_x+deltax1, cell1_y, cell1_y+deltay1, \
							rec["minx2"], rec["maxx2"], \
							rec["miny2"], rec["maxy2"])
				if (area_ov_mbr > 0.0):
					# print("j1: ",j1,"k1: ",k1,"area overlap MBR: ",area_ov_mbr)
					for j2 in range(DIM_H_X):
						for k2 in range(DIM_H_Y):
							cell2_x = rec["minx2"]+deltax2*k2
							cell2_y = rec["miny2"]+deltay2*j2
							area_ov = intersection(cell1_x, cell1_x+deltax1, cell1_y, cell1_y+deltay1, \
									cell2_x, cell2_x+deltax2, cell2_y, cell2_y+deltay2)
							if (area_ov > 0.0):
								# print("j1:",j1,"k1:",k1,"j2:",j2,"k2:",k2,"area overlap cells: ",area_ov)
								#
								# increment cardinality estimate
								# area of cells overlapping
								prob = ((hist_1_new[j1,k1,4]+hist_2_new[j1,k1,4])*(hist_1_new[j1,k1,5]+hist_2_new[j1,k1,5]))/area_ov
								n_prob_tot += 1
								if (prob > 1.0):
									prob = 1.0
									n_prob_uno += 1
								# for each cell of hist_1_new we consider the geometries inside the cell (128x128)
								# but for the hist_2_new we consider the geometries contained in the enlarged cell:
								# i.e. if ss=2 we consider 4 cells (j2,k2),(j2,k2+1),(j2+1,k2),(j2+1,k2+1)
								tot_geo1 = hist_1_new[j1,k1,0]
								tot_geo2 = 0
								for a in range(-step+1,step):
									for b in range(-step+1,step):
										if (j2+a < DIM_H_X and j2+a > 0 and k2+b > 0 and k2+b < DIM_H_Y):
											tot_geo2 += hist_2_new[j2+a,k2+b,0]
								new_jn_card += tot_geo1 * tot_geo2 * prob
								#int_cell[j2,k2] = hist_1_new[j1,k1,0] * hist_2_new[j2,k2,0] * prob
								
		#
		#print("int_cell.shape: ", int_cell.shape)
		#gh.prr_orig(int_cell)
		y = new_jn_card / (card1*card2)
		logger.debug("Old card/sel: "+str(rec["jn_card"])+"/"+str(out_y[i])+" New card/sel: "+str(new_jn_card)+"/"+str(y))
		logger.debug("Num prob equals to one: "+str(n_prob_uno)+"/"+str(n_prob_tot))
		out_y[i] = y
		out_jn_id[i] = rec['ds1']+"_e"+str(step)+"-"+rec['ds2']+"_e"+str(step)
		jn_res[out_jn_id[i]] = dict([('ds1', rec['ds1']+"_e"+str(step)),('distr1', rec['distr1']),('ds2', rec['ds2']+"_e"+str(step)),('distr2', rec['distr2']), \
                        ('card1', card1),('card2', card2), \
                        # ('deltax1', deltax1*factor),('deltax2', deltax2*factor), ('deltay1', deltay1*factor),('deltay2', deltay2*factor), \
                        ('avgLenX1', lenX1+deltax1*factor),('avgLenX2', lenX2+deltax2*factor), \
                        ('avgLenY1', lenY1+deltay1*factor),('avgLenY2', lenY2+deltay2*factor), \
                        ('avgArea1', (lenX1+deltax1*factor)*(lenY1+deltay1*factor) ),('avgArea2', (lenX2+deltax2*factor)*(lenY2+deltay2*factor)), \
                        ('minx1', rec["minx1"]), ('miny1', rec["miny1"]), ('maxx1', rec["maxx1"]), ('maxy1', rec["maxy1"]), \
                        ('minx2', rec["minx2"]), ('miny2', rec["miny2"]), ('maxx2', rec["maxx2"]), ('maxy2', rec["maxy2"]), \
                        ('jn_card', new_jn_card), ('jn_sel', y)])
		# SAVE HISTOGRAMS
		# 
		logger.debug("Saving first histogram...")
		fileHist1_new = pathHist + rec['ds1']+"_e" + str(step) + "_summary.csv"
		fieldnames = ['i0', 'i1', 'num_features', 'size', 'num_points', 'avg_area', 'avg_side_length_0', 'avg_side_length_1']
		with open(fileHist1_new, 'w', encoding='UTF8', newline='') as f:
			writer = csv.DictWriter(f, fieldnames=fieldnames)
			writer.writeheader()
			for ii in range(DIM_H_X):
				for jj in range(DIM_H_Y):
					if (hist_1_new[ii,jj,0] > 0.0):
						record = dict([('i0', ii), ('i1', jj), ('num_features', int(hist_1_new[ii,jj,0])), ('size', int(hist_1_new[ii,jj,1])), \
							('num_points', int(hist_1_new[ii,jj,2])), ('avg_area', hist_1_new[ii,jj,3]), \
							('avg_side_length_0', hist_1_new[ii,jj,4]), ('avg_side_length_1', hist_1_new[ii,jj,5])])
						writer.writerow(record)
		logger.debug("Saving second histogram...")
		fileHist2_new = pathHist + rec['ds2']+"_e" + str(step) + "_summary.csv"
		with open(fileHist2_new, 'w', encoding='UTF8', newline='') as f:
			writer = csv.DictWriter(f, fieldnames=fieldnames)
			writer.writeheader()
			for ii in range(DIM_H_X):
				for jj in range(DIM_H_Y):
					if (hist_2_new[ii,jj,0] > 0.0):
						record = dict([('i0', ii), ('i1', jj), ('num_features', int(hist_2_new[ii,jj,0])), ('size', int(hist_2_new[ii,jj,1])), \
							('num_points', int(hist_2_new[ii,jj,2])), ('avg_area', hist_2_new[ii,jj,3]), \
							('avg_side_length_0', hist_2_new[ii,jj,4]), ('avg_side_length_1', hist_2_new[ii,jj,5])])
						writer.writerow(record)

	logger.info("Enlargment end.")	
	return out_y, jn_res, out_jn_id

def intersection (minx1,maxx1,miny1,maxy1,minx2,maxx2,miny2,maxy2):
	if ((minx1 < maxx2) and (maxx1 > minx2)):
		if ((miny1 < maxy2) and (maxy1 > miny2)):
			# compute area
			a_x = np.array([minx1,minx2,maxx1,maxx2])
			a_y = np.array([miny1,miny2,maxy1,maxy2])
			a_x = np.sort(a_x)
			a_y = np.sort(a_y)
			area = (a_x[2] - a_x[1]) * (a_y[2] - a_y[1])
			return area
	return 0.0 

def gen_join_input_from_file_new(mode, flag_sel_card, from_x, to_x, local_enc, global_enc, resultFile, pathHist, delim, factor):
#
# PARAMETERS:
# mode: 0 the LOCAL and GLOBAL embeddings of both datasets are computed and added to x and xg respectively,
#       1 the LOCAL embeddings of both datasets are computed and added to x,
#         and the vector containing the MBR of both datasets are added to xg
# flag_sel_card: 0 stores in y selectivity,
#                1 cardinality,
#                2 mbrTests,
#                3 mbrTests selectivity
# local_enc: encoder for local embedding
# global_enc: encorder for global embedding
# resultFile: name of the file containing the results of the join operations
# pathHist: directory where the histograms are stored
# delim: delimiter of the resultFile
# 
# --------------------------------------------------------------------------
# Reading result file
#

	print("Reading the file with join results...")
	total = to_x - from_x
	out_y = np.zeros((total))
	out_x = np.zeros((total,DIM_E_X,DIM_E_Y,(2*DIM_E_Z)))
	out_distr = np.zeros((total,2))
	keep = np.ones((total), dtype=np.int8)

	if (mode == 0):
		out_xg = np.zeros((total,DIM_E_X,DIM_E_Y,(2*DIM_EG_Z)))
	else:
		out_xg = np.zeros((total, 8))

	with open(resultFile, mode='r') as csv_file:
		csv_reader = csv.DictReader(csv_file,delimiter=',')
		line_count = 0
		count = 0
		uniform_distr = 0
		for row in csv_reader:
			if (line_count == 0):
				print("Row number: "+str(line_count))
			if (line_count == to_x):
				break
			if (line_count < from_x):
				line_count += 1
				continue;

			file1 = row["ds1"]
			print("File1: ",file1)
			fileHist1 = pathHist + file1 + "_summary.csv"
			mbr1 = dict([('minx', float(row["minx1"])), ('miny', float(row["miny1"])), ('maxx', float(row["maxx1"])), ('maxy', float(row["maxy1"]))])
			file2 = row["ds2"]
			fileHist2 = pathHist + file2 + "_summary.csv"
			mbr2 = dict([('minx', float(row["minx2"])), ('miny', float(row["miny2"])), ('maxx', float(row["maxx2"])), ('maxy', float(row["maxy2"]))])

                	# computing X in the different cases
                	# mode == 0: local emb, global emb
			if (mode == 0):
				embL1, embG1 = get_embedding(local_enc, global_enc, fileHist1, mbr1)
				embL1 = embL1.numpy().reshape((DIM_E_X, DIM_E_Y, DIM_E_Z))
				embG1 = embG1.numpy().reshape((32, 32, 2))
				embL2, embG2 = get_embedding(local_enc, global_enc, fileHist2, mbr2)
				embL2 = embL2.numpy().reshape((DIM_E_X, DIM_E_Y, DIM_E_Z))
				embG2 = embG2.numpy().reshape((32, 32, 2))
				x = np.concatenate([embL1, embL2], axis=2)
				xg = np.concatenate([embG1, embG2], axis=2)
			# mode == 1: local emb and in a separate array mbr of both datasets
			else:
				embL1, embG1 = get_embedding(local_enc, global_enc, fileHist1, mbr1)
				embL1 = embL1.numpy().reshape((DIM_E_X, DIM_E_Y, DIM_E_Z))
				embG1 = embG1.numpy().reshape((32, 32, 2))
				embL2, embG2 = get_embedding(local_enc, global_enc, fileHist2, mbr2)
				embL2 = embL2.numpy().reshape((DIM_E_X, DIM_E_Y, DIM_E_Z))
				embG2 = embG2.numpy().reshape((32, 32, 2))
				x = np.concatenate([embL1, embL2], axis=2)
				xg = np.zeros((8))
				
				xg[0] = mbr1["minx"]
				xg[1] = mbr1["miny"]
				xg[2] = mbr1["maxx"]
				xg[3] = mbr1["maxy"]
				xg[4] = mbr2["minx"]
				xg[5] = mbr2["miny"]
				xg[6] = mbr2["maxx"]
				xg[7] = mbr2["maxy"]

			# computing Y in different cases
			# flag_sel_card == 0: selectivity
			if (flag_sel_card == 0):
				c1 = float(row["card1"])
				c2 = float(row["card2"])
				print("Factor: ",factor,"jn_card: ",float(row["jn_card"]))
				y = factor * float(row["jn_card"]) / (c1*c2)
				if (y >= 2.0):
					print("continue: ",str(line_count - from_x))
					keep[line_count - from_x] = 0
					#continue
				elif (y > 1.0):
					y = 1.0
			elif (flag_sel_card == 1):
				y = float(row["jn_card"])
			elif (flag_sel_card == 2):
				y = float(row["jn_NMBRTest"])
			else:
				c1 = row["card1"]
				c2 = row["card2"]
				y = factor * float(row["jn_NMBRTests"]) / (c1*c2)
				if (y >= 2):
					print("continue: ",str(line_count - from_x))
					keep[line_count - from_x] = 0
					#continue
				elif (y > 1.0):
					y = 1.0

			out_x[line_count - from_x] = x
			out_xg[line_count - from_x] = xg
			out_y[line_count-from_x] = y
		
			# storing distribution of datasets
			if (row["distr1"].lower() == "uniform"):
				d1 = 0
			elif (row["distr1"].lower() == "parcel"):
				d1 = 1
			elif (row["distr1"].lower() == "gaussian"):
				d1 = 2
			elif (row["distr1"].lower() == "bit"):
				d1 = 3
			elif (row["distr1"].lower() == "diagonal"):
				d1 = 4
			elif (row["distr1"].lower() == "sierpinski"):
				d1 = 5
			else:
				d1 = 6

			if (row["distr2"].lower() == "uniform"):
				d2 = 0
			elif (row["distr2"].lower() == "parcel"):
				d2 = 1
			elif (row["distr2"].lower() == "gaussian"):
				d2 = 2
			elif (row["distr2"].lower() == "bit"):
				d2 = 3
			elif (row["distr2"].lower() == "diagonal"):
				d2 = 4
			elif (row["distr2"].lower() == "sierpinski"):
				d2 = 5
			else:
				d2 = 6
			
			out_distr[line_count-from_x][0] = d1
			out_distr[line_count-from_x][1] = d2

			line_count += 1
			print("line: ", str(line_count))
	out_x = out_x[keep==1]
	out_xg = out_xg[keep==1]
	out_distr = out_distr[keep==1]
	out_y = out_y[keep==1]
	return out_x, out_xg, out_distr, out_y

def addNewPairByRotation(file_in,num_rot):
	# file_in: input file
	# num_rot: array containing the number of pairs to add for each of the 10 classes
	
	with open(file_in, mode='r') as csv_file:
		file_out = file_in[:-4] + "_rot.csv"
		csv_reader = csv.DictReader(csv_file,delimiter=',')
		line_count = 0
		rotation = 0
		with open(file_out, 'w', encoding='UTF8', newline='') as f:
			for row in csv_reader:
				if (line_count == 0):
					writer = csv.DictWriter(f, fieldnames=row)
					writer.writeheader()
					line_count += 1
					print("First row")
					continue
				if (int(row["Class"]) == 11):
					continue

				writer.writerow(row)
				for i in range(num_rot[int(row["Class"])]):
					row1 = row.copy()
					# row1["dataset1"] = row["dataset1"][0:-8]+"_r"+str(rotation)+row["dataset1"][-8:]
					# row1["dataset2"] = row["dataset2"][0:-8]+"_r"+str(rotation)+row["dataset2"][-8:]
					row1["dataset1"] = row["dataset1"][0:-4]+"_r"+str(rotation)+".csv"
					row1["dataset2"] = row["dataset2"][0:-4]+"_r"+str(rotation)+".csv"
					rotation += 1
					writer.writerow(row1)
	return 1

