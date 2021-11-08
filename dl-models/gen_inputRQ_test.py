#!/usr/bin/env python3
import os
from os import listdir, mkdir, path, sep
from os.path import isfile, join
import numpy as np
import random as rd
import math
import csv
import generate_histogram as gh
import generate_input_RQ as gi
#
def area_intersection(l1, r1, l2, r2):
        x = 0
        y = 1

        x_dist = (min(r1[x], r2[x]) -
                          max(l1[x], l2[x]))

        y_dist = (min(r1[y], r2[y]) -
                          max(l1[y], l2[y]))
        area = 0.0
        if (x_dist > 0.0 and y_dist > 0.0):
                area = x_dist * y_dist

        return area
#
def create_hist_rq(dimx, dimy, rq):
        X_MAX_G = 10
        X_MIN_G = 0
        Y_MAX_G = 10
        Y_MIN_G = 0
        sizeG_x = (X_MAX_G - X_MIN_G)/dimx
        sizeG_y = (Y_MAX_G - Y_MIN_G)/dimy
        cell_area = sizeG_x * sizeG_y

        out = np.zeros((dimx,dimy,1))

        start_cell_row = math.floor((rq['miny'] - Y_MIN_G)/sizeG_y)
        if (start_cell_row < 0):
                start_cell_row = 0
        if (start_cell_row > (dimy-1)):
                start_cell_row = dimy
        start_cell_col = math.floor((rq['minx'] - X_MIN_G)/sizeG_x)
        if (start_cell_col < 0):
                start_cell_col = 0
        if (start_cell_col > (dimx-1)):
                start_cell_col = dimx
        end_cell_row = math.floor((rq['maxy'] - Y_MIN_G)/sizeG_y)
        if (end_cell_row < 0):
                end_cell_row = -1
        if (end_cell_row > (dimy-1)):
                end_cell_row = (dimy-1)
        end_cell_col = math.floor((rq['maxx'] - X_MIN_G)/sizeG_x)
        if (end_cell_col < 0):
                end_cell_col = -1
        if (end_cell_col > (dimx-1)):
                end_cell_col = (dimx-1)
        print(str(start_cell_row),",",str(end_cell_row),"-",str(start_cell_col),",",str(end_cell_col))

        for i in range(start_cell_row, end_cell_row+1):
                for j in range(start_cell_col, end_cell_col+1):
                        cell_x_min = X_MIN_G + j * sizeG_x
                        cell_x_max = cell_x_min + sizeG_x
                        cell_y_min = Y_MIN_G + i * sizeG_y
                        cell_y_max = cell_y_min + sizeG_y
                        print("cell_x_min: ",cell_x_min, "cell_x_max: ", cell_x_max, "cell_y_min: ", cell_y_min,"cell_y_max: ", cell_y_max)
                        print("rq_minx: ", rq['minx'], "rq_miny", rq['miny'], "rq_maxx: ", rq['maxx'], "rq_maxy: ", rq['maxy'])
                        out[i,j] = area_intersection( (rq['minx'], rq['miny']) , (rq['maxx'], rq['maxy']), (cell_x_min, cell_y_min), (cell_x_max, cell_y_max) )/cell_area
                        print("cell_value: ",i," ",j, out[i,j])
        return out
#
#
def gen_rq_input_from_file(local_enc, global_enc, rqFile, mbrFile, resultFile, pathHist, delim, delim_res):
        # Reading RQ file
        rq = {}
        with open(rqFile, mode='r') as csv_file:
                csv_reader = csv.DictReader(csv_file,delimiter=',')
                line_count = 0
                rq_count = 0
                dName_old = ""
                for row1 in csv_reader:
                        if (line_count == 0):
                                print(f'Column names are: {", ".join(row1)}')
                                dName_old = row1["datasetName"]
                        print(f'\t{row1["datasetName"]},{row1["numQuery"]},{row1["queryArea"]}: {row1["minX"]}, {row1["minY"]}, {row1["maxX"]}, {row1["maxY"]}.')
                        dName = row1["datasetName"]
                        if (dName != dName_old):
                                rq_count = 0
                        name = dName+"-"+str(rq_count)
                        print(name)
                        rq[name] = dict([('minx', float(row1["minX"])), ('miny', float(row1["minY"])), ('maxx', float(row1["maxX"])), ('maxy', float(row1["maxY"]))])
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

        # Reading Result file
        with open(resultFile, mode='r') as csv_file:
                csv_reader = csv.DictReader(csv_file,delimiter=delim_res)
                line_count = 0
                for row in csv_reader:
                        line_count += 1
        line_count = 1000
        out_x = np.zeros((line_count,32,32,7))
        out_y = np.zeros((line_count))
        max_card = 0
        with open(resultFile, mode='r') as csv_file:
                csv_reader = csv.DictReader(csv_file,delimiter=delim_res)
                line_count = 0
                for row in csv_reader:
                        if (line_count == 0):
                                print(f'Column names are: {", ".join(row)}')

                        rq0 = rq[row["dataset"]+"-"+row["numQuery"]]
                        hist_RQ = create_hist_rq(128, 128, rq0)
                        fileHist = pathHist+"/"+row["dataset"] + "_summary.csv"
                        embL, embG, embRQ = gi.get_embedding(local_enc, global_enc, hist_RQ, fileHist, mbr[row["dataset"]])
                        print(embL.shape)
                        print(embG.shape)
                        print(embRQ.shape)
                        embL = embL.numpy().reshape((32,32,3))
                        embG = embG.numpy().reshape((32,32,2))
                        embRQ = embRQ.numpy().reshape((32,32,2))

                        x = np.concatenate([embL,embG,embRQ], axis=2)

                        out_x[line_count] = x
                        out_y[line_count] = float(row["cardinality"])
                        if (out_y[line_count] > max_card):
                                max_card = out_y[line_count]
                        line_count += 1
                        print("line: ",str(line_count))
                        if (line_count == 1000):
                                out_y = gh.nor_with_min_max(out_y,1,0.0,max_card)
                                return out_x, out_y
        out_y = gh.nor_with_min_max(out_y,1,0.0,max_card)
        return out_x, out_y
