#!/usr/bin/env python3
import os
from os import listdir, mkdir, path, sep
from os.path import isfile, join
import numpy as np
import random as rd
import math
import csv
import generate_histogram as gh

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
NORM_MAX = [8.77805800e+06, 3.05404802e+09, 1.53571255e+08, 3.03019291e-02, 1.91233400e-01, 2.20753674e-01]

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
	hist_local_norm = gh.nor_g_ab(hist_local.reshape((1,DIM_H_X,DIM_H_Y,DIM_H_Z)), 1, NORM_MIN, NORM_MAX)
	emb_local = local_enc.encoder(hist_local_norm.reshape((1,DIM_H_X,DIM_H_Y,DIM_H_Z)))

	# computing global histogram ----------------------
	hist_glob = gh.gen_global_hist(hist_local, DIM_H_X, DIM_H_Y, mbr)
	hist_glob_norm = gh.nor_g_ab(hist_glob.reshape((1,DIM_H_X,DIM_H_Y)), 1, NORM_MIN_G, NORM_MAX_G)
	emb_global = global_enc.encoder(hist_glob_norm.reshape((1,DIM_H_X,DIM_H_Y,DIM_HG_Z)))

	return emb_local, emb_global

def gen_join_input_from_file(mode, flag_sel_card, from_x, to_x, local_enc, global_enc, mbrFile, resultFile, pathHist, delim):
#
# PARAMETERS:
# flag_sel_card: 0 stores in y the selectivity, 1 the cardinality, 2 the mbrTests
# mode: 0 the LOCAL and GLOBAL embeddings of both datasets are computed and added to x and xg respectively, 
#       1 the LOCAL embeddings of both datasets are computed and added to x, 
#         and the vector containing the MBR of both datasets are added to xg
# local_enc: encoder for local embedding
# global_enc: encorder for global embedding
# mbrFile: name of the file containing the mbr of the datasets (datasetSummary)
# resultFile: name of the file containing the results of the join operations
# pathHist: directory where the histograms are stored
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
			print(f'\t{row["dataset"]},{row["distribution"]}: {row["x1"]}, {row["y1"]}, {row["x2"]}, {row["y2"]}.')
			name = row["dataset"]
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
			idx = file1.find("dataset-")
			idx_pnt = file1.find(".")
			file1 = file1[idx:idx_pnt]
			fileHist1 = pathHist + file1 + "_summary.csv"
			file2 = row["dataset2"]
			idx = file2.find("dataset-")
			idx_pnt = file2.find(".")
			file2 = file2[idx:idx_pnt]
			fileHist2 = pathHist + file2 + "_summary.csv"

                	# computing X in the different cases
                	# mode == 0: local emb, global emb
			if (mode == 0):
				embL1, embG1 = get_embedding(local_enc, global_enc, fileHist1, mbr[file1])
				embL1 = embL1.numpy().reshape((32, 32, 3))
				embG1 = embG1.numpy().reshape((32, 32, 2))
				embL2, embG2 = get_embedding(local_enc, global_enc, fileHist2, mbr[file2])
				embL2 = embL2.numpy().reshape((32, 32, 3))
				embG2 = embG2.numpy().reshape((32, 32, 2))
				x = np.concatenate([embL1, embL2], axis=2)
				xg = np.concatenate([embG1, embG2], axis=2)
			# mode == 1: local emb and in a separate array mbr of both datasets
			else:
				embL1, embG1 = get_embedding(local_enc, global_enc, fileHist1, mbr[file1])
				embL1 = embL1.numpy().reshape((32, 32, 3))
				embG1 = embG1.numpy().reshape((32, 32, 2))
				embL2, embG2 = get_embedding(local_enc, global_enc, fileHist2, mbr[file2])
				embL2 = embL2.numpy().reshape((32, 32, 3))
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

			out_x[line_count - from_x] = x
			out_xg[line_count - from_x] = xg

			# computing Y in different cases
			# flag_sel_card == 0: selectivity
			if (flag_sel_card == 0):
				c1 = features[file1]["card"]
				c2 = features[file2]["card"]
				y = float(row["resultSJSize"]) / (c1*c2)
			elif (flag_sel_card == 1):
				y = float(row["resultSJSize"])
			else:
				y = float(row["PBSMMBRTests"])
                
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

			out_distr[line_count-from_x][0] = d1
			out_distr[line_count-from_x][1] = d2

			line_count += 1
			print("line: ", str(line_count))

	return out_x, out_xg, out_distr, out_y
