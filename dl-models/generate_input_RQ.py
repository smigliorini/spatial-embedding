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

NORM_MIN_G = [0.0]
NORM_MAX_G = [8708693.144550692]

X_MIN_REF = 0
X_MAX_REF = 10
Y_MIN_REF = 0
Y_MAX_REF = 10

# -------------------------------

def get_embedding(local_enc, global_enc, rq_hist, datasetFile, mbr):
    # get local histogram
    hist_local = gh.gen_hist_from_file(DIM_H_X,DIM_H_Y,DIM_H_Z,datasetFile)
    hist_local_norm = gh.nor_a_ab(hist_local.reshape((1,DIM_H_X,DIM_H_Y,DIM_H_Z)), 1, NORM_MIN, NORM_MAX)
    emb_local = local_enc.encoder(hist_local_norm.reshape((1,DIM_H_X,DIM_H_Y,DIM_H_Z)))

    # computing global histogram
    hist_glob = gh.gen_global_hist(hist_local, DIM_H_X, DIM_H_Y, mbr)
    hist_glob_norm = gh.nor_g_ab(hist_glob.reshape((1,DIM_H_X,DIM_H_Y,DIM_HG_Z)), 1, NORM_MIN_G, NORM_MAX_G)
    emb_global = global_enc.encoder(hist_glob_norm.reshape((1,DIM_H_X,DIM_H_Y,DIM_HG_Z)))

    # computing embedding of rq_histogram
    emb_rq = global_enc.encoder(rq_hist.reshape((1, DIM_H_X, DIM_H_Y, DIM_HG_Z)))

    return emb_local, emb_global, emb_rq

def gen_rq_input_from_file(local_enc, global_enc, rqFile, mbrFile, resultFile, cardFile, pathHist, delim):
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
                print(
                    f'\t{row1["datasetName"]},{row1["numQuery"]},{row1["queryArea"]}: {row1["minX"]}, {row1["minY"]}, {row1["maxX"]}, {row1["maxY"]}.')
                dName = row1["datasetName"]
                if (dName != dName_old):
                    rq_count = 0
                name = dName + "-" + str(rq_count)
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
                        print(f'\t{row["datasetName"]},{row["Collection"]}: {row["minX"]}, {row["minY"]}, {row["maxX"]}, {row["maxY"]}.')
                        name = row["datasetName"]
                        mbr[name] = dict([('minx', float(row["minX"])), ('miny', float(row["minY"])), ('maxx', float(row["maxX"])), ('maxy', float(row["maxY"]))])
                        line_count += 1

        # Reading Cardinality of datasets
        card = {}
        with open(cardFile, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter=delim)
            line_count = 0
            for row in csv_reader:
                if (line_count == 0):
                    print(f'Column names are: {", ".join(row)}')
                print(f'\t{row["dataset"]}: {row["num_features"]}.')
                name = row["dataset"]
                card[name] = dict([('numFeatures', float(row["num_features"]))])
                line_count += 1

        # Reading Result file
        with open(resultFile, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file,delimiter=delim)
            line_count = 0
            for row in csv_reader:
                line_count += 1
        out_x = np.zeros((line_count,DIM_E_X,DIM_E_Y,(DIM_E_Z + 2*DIM_EG_Z)))
        out_y = np.zeros((line_count))
        with open(resultFile, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file,delimiter=delim)
            line_count = 0
            count = 0
            max_selectivity = 0.5
            for row in csv_reader:
                if (line_count == 0):
                    print(f'Column names are: {", ".join(row)}')
                fileHist = pathHist + row["dataset"] + "_summary.csv"
                rq0 = rq[row["dataset"] + "-" + row["numQuery"]]
                hist_RQ = gen_rq_layer(rq0, DIM_H_X, DIM_H_Y)
                embL, embG, embRQ = get_embedding(local_enc, global_enc, hist_RQ, fileHist, mbr[row["dataset"]])
                # embL, embG, embRQ = gi.get_embedding(local_enc, global_enc, fileHist, mbr[row["dataset"]])
                #print(embL.shape)
                #print(embG.shape)
                #print(embRQ.shape)
                embL = embL.numpy().reshape((32, 32, 3))
                embG = embG.numpy().reshape((32, 32, 2))
                embRQ = embRQ.numpy().reshape((32, 32, 2))

                x = np.concatenate([embL, embG, embRQ], axis=2)

                out_x[line_count] = x

                c = card[row["dataset"]]
                out_y[line_count] = float(row["cardinality"]) / c["numFeatures"]

                line_count += 1
                print("line: ", str(line_count))
                #if (line_count == 10):
                #    out_y = gh.nor_with_min_max(out_y, 100, 0.0, max_selectivity)
                #    return out_x, out_y
            out_y = gh.nor_with_min_max(out_y, 100, 0.0, max_selectivity)
            return out_x, out_y

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

