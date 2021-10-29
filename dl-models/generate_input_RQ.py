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

NORM_MIN = [0. 0. 0. 0. 0. 0.]
NORM_MAX = [8.77805800e+06 3.05404802e+09 1.53571255e+08 3.03019291e-02 1.91233400e-01 2.20753674e-01]

NORM_MIN_G = 0.0
NORM_MAXI_G = 8708693.144550692

X_MIN_REF = 0
X_MAX_REF = 10
Y_MIN_REF = 0
Y_MAX_REF = 10

# -------------------------------

def get_embedding(local_enc, global_enc, datasetFile, mbr)
	# get local histogram
	hist_local = gh.gen_hist_from_file(DIM_H_X,DIM_H_Y,DIM_H_Z,datasetFile)
	hist_local_norm = gh.nor_with_min_max(hist_local, NORM_MIN, NORM_MAX)
	emb_local = local_enc.encoder(hist_local_norm.reshape((1,DIM_H_X,DIM_H_Y,DIM_H_Z)))
	
	# computing globla histogram
	hist_glob = gh.gen_global_hist(hist_local, DIM_H_X, DIM_H_Y, mbr)
	hist_glob_norm = gh.nor_with_min_max(hist_glob, NORM_MIN_G, NORM_MAX_G)
	emb_global = global_enc.encoder(hist_glob_norm.reshape((1,DIM_H_X,DIM_H_Y,DIM_HG_Z))) 

	return emb_local, emb_global

def gen_rq_input_from_file(local_enc, global_enc, rqFile, mbrFile, resultFile, pathHist, del):
# local_enc: encoder for local embedding
# global_enc: encorder for global embedding
# rqFile: the name of the file containing the MBR of the queries
#
        # Reading RQ file
        rq = {}
        with open(rqFile, mode='r') as csv_file:
                csv_reader = csv.DictReader(csv_file,delimiter=del)
                line_count = 0
                for row in csv_reader:
                        if (line_count == 0):
                                print(f'Column names are: {", ".join(row)}')
                        print(f'\t{row["datasetName"]},{row["numQuery"]}: {row["minX"]}, {row["minY"]}, {row["maxX"]}, {row["maxY"]}.')
                        name = row["datasetName"]+"-"+row["numQuery"]
                        rq[name] = dict([('minx', float(row["minX"])), ('miny', float(row["minY"])), ('maxx', float(row["maxX"])), ('maxy', float(row["maxY"]))])
                        line_count += 1

	# Reading MBR file
	mbr = {}
        with open(mbrFile, mode='r') as csv_file:
                csv_reader = csv.DictReader(csv_file,delimiter=del)
                line_count = 0
                for row in csv_reader:
                        if (line_count == 0):
                                print(f'Column names are: {", ".join(row)}')
                        print(f'\t{row["datasetName"]},{row["Collection"]}: {row["minX"]}, {row["minY"]}, {row["maxX"]}, {row["maxY"]}.')
                        name = row["datasetName"]
                        mbr[name] = dict([('minx', float(row["minX"])), ('miny', float(row["minY"])), ('maxx', float(row["maxX"])), ('maxy', float(row["maxY"]))])
                        line_count += 1

	# Reading Result file
	with open(resultFile, mode='r') as csv_file:
                csv_reader = csv.DictReader(csv_file,delimiter=del)
                line_count = 0
		for row in csv_reader:
			line_count += 1	
	out_x = np.zeros((line_count,dimx,dimy,dimz))
	out_y = np.zeros((line_count))
	with open(resultFile, mode='r') as csv_file:
                csv_reader = csv.DictReader(csv_file,delimiter=del)
                line_count = 0
                for row in csv_reader:
			if (line_count == 0):
                                print(f'Column names are: {", ".join(row)}')
			
			fileHist = pathHist+row["datasetName"] + "_summary.csv"
			embL, embG = get_embedding(local_enc, global_enc, fileHist, mbr[row["datasetName"]])

			# computing histogram for range query
			rq0 = rq[row["datasetName"]+"-"+row["numQuery"]]
			embRQ = create_hist_rq(DIM_E_X, DIM_E_Y,rq0)

			# "unione" dei tre embedding
			a = unione
			out_x[count] = a
			out_y[count] = row["..."]
			count += 1
        return out_x, out_y

