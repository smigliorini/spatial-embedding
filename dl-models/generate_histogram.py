#!/usr/bin/env python3
import os
from os import listdir, mkdir, path, sep
from os.path import isfile, join
import numpy as np
import random as rd
import math
import csv

GLOBAL_X_MIN = 0
GLOBAL_Y_MIN = 0
GLOBAL_X_MAX = 10
GLOBAL_Y_MAX = 10

X_MIN = 0
Y_MIN = 0
X_MAX = 128
Y_MAX = 128

SIZE = 128


def count_frequency_1(a):
    freq = np.zeros(11)
    for i in range(a.shape[0]):
        if (a[i] == 0):
            index = 0
        else:
            index = math.ceil(a[i] * 10)
        if (index >= 11):
            index = 10
        freq[index] += 1
    return freq


def count_frequency(a, dimx, dimy):
    freq = np.zeros(11)
    for i in range(a.shape[0]):
        for j in range(dimx):
            for k in range(dimy):
                if (a[i, j, k] == 0):
                    index = 0
                else:
                    index = math.ceil(a[i, j, k] * 10)
                if (index >= 11):
                    index = 10
                freq[index] += 1
    return freq


def count_frequency_std(a, dimx, dimy):
    freq = np.zeros(22)
    for i in range(a.shape[0]):
        for j in range(dimx):
            for k in range(dimy):
                index = math.floor(a[i, j, k] * 10)
                index += 10
                if (index < 0):
                    index = 0
                if (index > 20):
                    index = 21
                freq[index] += 1
    return freq


# composite explorer
def get_files_path(path: str):
    list_files_paths = []
    for structure in listdir(path):
        sub_path = join(path, structure)
        if isfile(sub_path):
            list_files_paths.append(sub_path)
        else:
            list_files_paths = list_files_paths + get_files_path(sub_path)
    return list_files_paths


# generate one histogram from one file of Ahmed
def gen_hist_from_file(dimx, dimy, dimz, file):
    h0 = np.zeros((dimx, dimy, dimz))
    with open(file, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            # DEBUG print(f'Column names are: {", ".join(row)}')

            # DEBUG print(f'\t{row["i0"]},{row["i1"]}: {row["num_features"]}, {row["size"]}, {row["num_points"]}, {row["avg_area"]}, {row["avg_side_length_0"]}, {row["avg_side_length_1"]}.')
            x = int(row["i0"])
            y = int(row["i1"])
            if (dimz >= 1):
                h0[x, y, 0] = int(row["num_features"])
            if (dimz >= 2):
                h0[x, y, 1] = int(row["size"])
            if (dimz >= 3):
                h0[x, y, 2] = int(row["num_points"])
            if (dimz >= 4):
                h0[x, y, 3] = float(row["avg_area"])
            if (dimz >= 5):
                h0[x, y, 4] = float(row["avg_side_length_0"])
            if (dimz >= 6):
                h0[x, y, 5] = float(row["avg_side_length_1"])
            line_count += 1
        print(f'Processed {line_count} lines.')
    return h0


#
# loading local histograms and generating global histograms
#
def gen_input_from_file(dimx, dimy, dimz, path, mbrFile, suffix):
    #
    # path: directory where the files containing the histograms are located: for example histograms_small
    # mbrFile: the name of the file containing the MBR of the datasets the histograms refer to
    # suffix: the suffix that must to be added in order to obtain from the name of the file the name of the dataset: for example '_s'
    #
    # Reading MBR file
    mbr = {}
    with open(mbrFile, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if (line_count == 0):
                print(f'Column names are: {", ".join(row)}')
            print(
                f'\t{row["datasetName"]},{row["Collection"]}: {row["minX"]}, {row["minY"]}, {row["maxX"]}, {row["maxY"]}.')
            name = row["datasetName"]
            mbr[name] = dict([('minx', float(row["minX"])), ('miny', float(row["minY"])), ('maxx', float(row["maxX"])),
                              ('maxy', float(row["maxY"]))])
            line_count += 1

    # rq = {}
    # with open(rqFile, mode='r') as csv_file:
    #	csv_reader = csv.DictReader(csv_file)
    #	line_count = 0
    #	for row in csv_reader:
    #		rq[row["dataset"]+"_"+row["numQuery"]] = {"cardinality": row["cardinality"], "minx": row["minx"], "maxx": row["maxx"], "miny": row["miny"], "maxy": row["maxy"]}

    files = get_files_path(path)
    print('Found {0} files'.format(len(files)))
    hh = np.zeros((len(files), dimx, dimy, dimz))
    hg = np.zeros((len(files), dimx, dimy))
    count = 0
    # for each file, generate one local histogram and one global histogram
    for ff in files:
        print('Processing file {0} ...'.format(ff))
        h0 = gen_hist_from_file(dimx, dimy, dimz, ff)
        hh[count] = h0
        # searching MBR
        name = (ff.rpartition("/")[2]).rpartition("_summary")[0] + suffix
        print('Searching mbr for ' + name)
        mbr0 = mbr[name]
        print('Find: ', mbr0)
        # computing global histogram
        hg[count] = gen_global_hist(h0, dimx, dimy, mbr0)
        count += 1
    return hh, hg


def gen_global_hist(h0, dimx, dimy, mbr):
    # the dimensions of the grid (dimx,dimy) of the local and global histograms are the same
    #	xsize = (X_MAX - X_MIN) / SIZE
    xsize = (mbr['maxx'] - mbr['minx']) / dimx
    #	ysize = (Y_MAX - Y_MIN) / SIZE
    ysize = (mbr['maxy'] - mbr['miny']) / dimy
    # print('Cell sides: ',xsize," x ",ysize)

    cellArea = xsize * ysize

    xsizeG = (GLOBAL_X_MAX - GLOBAL_X_MIN) / dimx
    ysizeG = (GLOBAL_Y_MAX - GLOBAL_Y_MIN) / dimy
    # print('Global Cell sides: ',xsizeG," x ",ysizeG)

    hg = np.zeros((dimx, dimy))

    # card = num_features == 0

    for i in range(dimx):
        for j in range(dimy):
            cell = h0[i, j]
            if (cell[0] == 0):
                continue;
            xC = mbr['minx'] + xsize * j
            yC = mbr['miny'] + ysize * i
            # print('Cell coord: (',xC,',',yC,',',xC+xsize,',',yC+ysize,')')
            firstCellGcol = math.floor(xC / xsizeG)
            firstCellGrow = math.floor(yC / ysizeG)
            # print('Global Cell coord: (',firstCellGrow,',',firstCellGcol,')')
            # print('Cell intersection: ',area_intersection((xC, yC), (xC + xsize, yC + ysize), (firstCellGcol * xsizeG, firstCellGrow * ysizeG), (firstCellGcol * xsizeG + xsizeG, firstCellGrow * ysizeG + ysizeG)))
            hg[firstCellGrow, firstCellGcol] += (cell[0] * area_intersection((xC, yC), (xC + xsize, yC + ysize), (
                firstCellGcol * xsizeG, firstCellGrow * ysizeG), (firstCellGcol * xsizeG + xsizeG,
                                                                  firstCellGrow * ysizeG + ysizeG)) / cellArea)
            # cell[3])

            secondCellGcol = math.floor((xC + xsize) / xsizeG)
            if secondCellGcol > firstCellGcol:
                hg[firstCellGrow, secondCellGcol] += (cell[0] * area_intersection((xC, yC), (xC + xsize, yC + ysize), (
                    secondCellGcol * xsizeG, firstCellGrow * ysizeG), (secondCellGcol * xsizeG + xsizeG,
                                                                       firstCellGrow * ysizeG + ysizeG)) / cellArea)

            secondCellGrow = math.floor((yC + ysize) / ysizeG)
            if secondCellGrow > firstCellGrow:
                hg[secondCellGrow, firstCellGcol] += (cell[0] * area_intersection((xC, yC), (xC + xsize, yC + ysize), (
                    firstCellGcol * xsizeG, secondCellGrow * ysizeG), (firstCellGcol * xsizeG + xsizeG,
                                                                       secondCellGrow * ysizeG + ysizeG)) / cellArea)

            if secondCellGrow > firstCellGrow and secondCellGcol > firstCellGcol:
                hg[secondCellGrow, secondCellGcol] += (cell[0] * area_intersection((xC, yC), (xC + xsize, yC + ysize), (
                    secondCellGcol * xsizeG, secondCellGrow * ysizeG), (secondCellGcol * xsizeG + xsizeG,
                                                                        secondCellGrow * ysizeG + ysizeG)) / cellArea)
    return hg


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


def gen_rq_from_file(dimx, dimy, path):
    files = get_files_path(path)
    print('Found {0} files'.format(len(files)))
    r = np.zeros((len(files), dimx, dimy))
    b = np.zeros((len(files)))
    count = 0
    # TODO
    return r, b


def cell_mod(dimz, card, delta, lenx, dx, leny, dy, area, dia, pnt, deltapnt, size, deltasize):
    if (dimz == 1):
        cloc = 0.0
    else:
        cloc = np.zeros(dimz)
    if (dimz > 0):
        if (dimz == 1):
            cloc = card + rd.randint(-delta, delta)
        else:
            cloc[0] = card + rd.randint(-delta, delta)
    if (dimz > 1):
        cloc[1] = pnt + rd.randint(-deltapnt, deltapnt)
    if (dimz > 2):
        cloc[2] = size + rd.randint(-deltasize, deltasize)
    if (dimz > 3):
        cloc[3] = 0.000001 + lenx + rd.randint(-1, 1) * rd.random() * dx
    if (dimz > 4):
        cloc[4] = 0.000001 + leny + rd.randint(-1, 1) * rd.random() * dy
    if (dimz > 5):
        cloc[5] = 0.0000001 + area + rd.randint(-1, 1) * rd.random() * dx * dy
    return cloc


def generate(num, dimx, dimy, dimz, type):
    debug = 0
    if (dimz > 6):
        return "Max value for dimz is 6"
    if (dimz == 1):
        a = np.zeros((num, dimx, dimy))
    else:
        a = np.zeros((num, dimx, dimy, dimz))
    for i in range(num):
        if ((i % math.ceil(num / 10)) == 0):
            print("Done: ", i, "/", num)
        card = rd.randint(10, 1000)
        delta = math.ceil(0.2 * card)
        lenx = rd.random() * 0.2
        dx = 0.1 * lenx
        leny = rd.random() * 0.2
        dy = 0.1 * leny
        area = lenx * leny
        dia = math.ceil(dimx / 5) * rd.random()
        pnt = rd.randint(50, 100000)
        deltapnt = math.ceil(0.1 * pnt)
        size = rd.randint(100, 10000)
        deltasize = math.ceil(0.1 * size)
        vert = math.ceil(dimy * rd.random())
        diav = math.ceil(dimy / 5) * rd.random()
        horiz = math.ceil(dimx * rd.random())
        diah = math.ceil(dimx / 5) * rd.random()
        for j in range(dimx):
            for k in range(dimy):
                if (type == 0):  # UNIFORM DISTRIBUTION
                    a[i, j, k] = cell_mod(dimz, card, delta, lenx, dx, leny, dy, area, dia, pnt, deltapnt, size,
                                          deltasize)
                if (type == 1 and abs(j - k) < dia):  # DIAGONAL DISTRIBUTION
                    a[i, j, k] = cell_mod(dimz, card, delta, lenx, dx, leny, dy, area, dia, pnt, deltapnt, size,
                                          deltasize)
                if (type == 2 and (abs(k - vert) < diaiv)):  # VERTICAL DISTR
                    a[i, j, k] = cell_mod(dimz, card, delta, lenx, dx, leny, dy, area, dia, pnt, deltapnt, size,
                                          deltasize)
                if (type == 3 and (abs(j - horiz) < diah)):  # HORIZONTAL DISTR
                    a[i, j, k] = cell_mod(dimz, card, delta, lenx, dx, leny, dy, area, dia, pnt, deltapnt, size,
                                          deltasize)
                if (type == 4 and (abs(j - horiz) < diah) and (abs(k - vert) < diav)):  # GAUSSIAN DISTR
                    a[i, j, k] = cell_mod1(dimz, card, delta, lenx, dx, leny, dy, area, dia, pnt, deltapnt, size,
                                           deltasize)
                if (type == 5):  # MIX
                    if (i % 10 in [0, 2, 6, 8]) or \
                            (i % 10 in [1, 4] and (abs(j - k) < dia)) or \
                            (i % 10 == 3 and (abs(k - vert) < diav)) or \
                            (i % 10 == 5 and (abs(j - horiz) < diah)) or \
                            (i % 10 in [7, 9] and (abs(j - horiz) < diah) and (abs(k - vert) < diav)):
                        a[i, j, k] = cell_mod(dimz, card, delta, lenx, dx, leny, dy, area, dia, pnt, deltapnt, size,
                                              deltasize)
    if (debug == 1):
        for i in range(num):
            count = 0
            count_zero = 0
            for j in range(dimx):
                for k in range(dimy):
                    count = count + 1
                    if a[i, j, k, 0] == 0.0:
                        count_zero = count_zero + 1
            print("Riga: ", i, " -> ", count_zero, "/", count)

    print("End generation of local histograms")
    print("Generating global histograms...")
    g = np.zeros((num, dimx, dimy))
    for i in range(num):
        j = rd.randint(0, math.ceil(dimx * 2 / 3))
        k = rd.randint(0, math.ceil(dimy * 2 / 3))
        d_j = rd.randint(math.ceil(dimx / 6), math.ceil(dimx / 2))
        d_k = rd.randint(math.ceil(dimy / 6), math.ceil(dimy / 2))
        card = rd.randint(100, 8000)
        delta = math.ceil(0.2 * card)
        for jbis in range(d_j):
            for kbis in range(d_k):
                if (j + jbis < dimx and k + kbis < dimy):
                    g[i, j + jbis, k + kbis] = card + rd.randint(-delta, delta)

    print("Generating random range query selectivities...")
    b = np.zeros(num)
    for i in range(num):
        if (rd.random() < 0.8):
            b[i] = rd.randint(0, 1200)

    print("Generating range queries...")
    r = np.zeros((num, dimx, dimy))
    for i in range(num):
        j = rd.randint(0, dimx - 1)
        k = rd.randint(0, dimy - 1)
        d_j = rd.randint(1, math.ceil(dimx / 25))
        d_k = rd.randint(1, math.ceil(dimy / 25))
        for jbis in range(d_j):
            for kbis in range(d_k):
                if (j + jbis < dimx and k + kbis < dimy):
                    r[i, j + jbis, k + kbis] = 1.0
    print("Returning a, g, b and r")
    return a, g, b, r


def prr(a, scale):
    for i in range(a.shape[0] - 1, -1, -1):
        for j in range(a.shape[1]):
            v = int(a[i, j] * scale) % 10
            print(v, end='')
        print("#")


def prr_orig(a):
    for i in range(a.shape[0] - 1, -1, -1):
        for j in range(a.shape[1]):
            v = int(a[i, j]) % 10
            print(v, end='')
        print("#")


def prr_delta10(a, da):
    for i in range(a.shape[0] - 1, -1, -1):
        for j in range(a.shape[1]):
            v = int(abs(a[i, j] - da[i, j]) * 10) % 10
            print(v, end='')
        print("#")


def prr_delta100(a, da):
    avg = 0
    for i in range(a.shape[0] - 1, -1, -1):
        avg = 0
        for j in range(a.shape[1]):
            v = int(abs(a[i, j] - da[i, j]) * 100) % 10
            avg = avg + v
            print(v, end='')
        print("#", int(avg) / a.shape[1])


def prr1(a, s1, s2):
    for i in range(a.shape[0] - 1, -1, -1):
        for j in range(a.shape[1]):
            if (a[i, j] <= s1):
                print("0", end='')
            else:
                if (a[i, j] <= s2):
                    print("-", end='')
                else:
                    print("1", end='')
        print("#")


def prr_a(a, f, s1, s2):
    for i in range(a.shape[0] - 1, -1, -1):
        for j in range(a.shape[1]):
            if (a[i, j, f] <= s1):
                print("0", end='')
            else:
                if (a[i, j, f] <= s2):
                    print("-", end='')
                else:
                    print("1", end='')
        print("#")


def std_a(a, zero):
    if (zero == 0):  # zero=0 includes the zeros in the average
        print("Standardizing 'a' including zeros...")
    else:  # zero=1 excludes the zeros in the average
        print("Standardizing 'a' without zeros...")
    if (a.ndim == 4):
        avg = np.zeros((a.shape[3]))
        count_zero = np.zeros((a.shape[3]))
        den = np.ones((a.shape[3]))
    else:
        avg = 0
        count_zero = 0
        den = 1
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            for k in range(a.shape[2]):
                if (a.ndim == 4):
                    for l in range(a.shape[3]):
                        if (a[i, j, k, l] == 0):
                            count_zero[l] += 1
                        else:
                            avg[l] = avg[l] + a[i, j, k, l]
                else:
                    if (a[i, j, k] == 0):
                        count_zero += 1
                    else:
                        avg = avg + a[i, j, k]

    if (a.ndim == 4):
        for l in range(a.shape[3]):
            den[l] = a.shape[0] * a.shape[1] * a.shape[2]
    else:
        den = a.shape[0] * a.shape[1] * a.shape[2]
    if (zero == 1):
        if (a.ndim == 4):
            for l in range(a.shape[3]):
                den[l] = den[l] - count_zero[l]
        else:
            den = den - count_zero
    if (a.ndim == 4):
        for l in range(a.shape[3]):
            avg[l] = avg[l] / den[l]
    else:
        avg = avg / den
    if (a.ndim == 4):
        mse = np.zeros((a.shape[3]))
    else:
        mse = 0
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            for k in range(a.shape[2]):
                if (a.ndim == 4):
                    if (zero == 0 or a[i, j, k, l] > 0):
                        for l in range(a.shape[3]):
                            mse[l] = mse[l] + (avg[l] - a[i, j, k, l]) ** 2
                else:
                    if (zero == 0 or a[i, j, k] > 0):
                        mse = mse + (avg - a[i, j, k]) ** 2
    if (a.ndim == 4):
        for l in range(a.shape[3]):
            mse[l] = pow(mse[l] / den[l], 0.5)
    else:
        mse = pow(mse / den, 0.5)

    if (a.ndim == 4):
        scal_a = np.zeros((a.shape[0], a.shape[1], a.shape[2], a.shape[3]))
    else:
        scal_a = np.zeros((a.shape[0], a.shape[1], a.shape[2]))

    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            for k in range(a.shape[2]):
                if (a.ndim == 4):
                    for l in range(a.shape[3]):
                        if (zero == 0 or a[i, j, k, l] > 0):
                            scal_a[i, j, k, l] = (a[i, j, k, l] - avg[l]) / mse[l]
                else:
                    if (zero == 0 or a[i, j, k] > 0):
                        scal_a[i, j, k] = (a[i, j, k] - avg) / mse
    print("avg and mse of the local histograms")
    print(avg, mse)
    return scal_a


def std_g(g):
    print("Standadizing g...")
    scal_g = np.zeros((g.shape[0], g.shape[1], g.shape[2]))
    avg = 0
    mse = 0
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            for k in range(g.shape[2]):
                avg = avg + g[i, j, k]
    den = g.shape[0] * g.shape[1] * g.shape[2]
    avg = avg / den
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            for k in range(g.shape[2]):
                mse = mse + (avg - g[i, j, k]) ** 2
    mse = pow(mse / den, 0.5)

    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            for k in range(g.shape[2]):
                scal_g[i, j, k] = (g[i, j, k] - avg) / mse
    print("avg and mse of the global histograms")
    print(avg, mse)
    return scal_g


def std_b(b):
    print("Standadizing b...")
    scal_b = np.zeros(b.shape[0])
    avg = 0
    mse = 0
    for i in range(b.shape[0]):
        avg = avg + b[i]
    avg = avg / b.shape[0]
    for i in range(b.shape[0]):
        mse = mse + (avg - b[i]) ** 2
    mse = pow(mse / b.shape[0], 0.5)

    for i in range(b.shape[0]):
        scal_b[i] = (b[i] - avg) / mse

    print("avg and mse of the selectivity of range queries")
    print(avg, mse)

    return scal_b


def shift_pos(a):
    print("Shifting a...")
    if (a.ndim == 4):
        min = np.ones(a.shape[3]) * 10000000000
    else:
        min = 10000000000
    print("Computing min...")
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            for k in range(a.shape[2]):
                if (a.ndim == 4):
                    for l in range(a.shape[3]):
                        if (a[i, j, k, l] < min[l]):
                            min[l] = a[i, j, k, l]
                else:
                    if (a[i, j, k] < min):
                        min = a[i, j, k]
    print("min: ", min)
    if (a.ndim == 4):
        shift_a = np.zeros((a.shape[0], a.shape[1], a.shape[2], a.shape[3]))
    else:
        shift_a = np.zeros((a.shape[0], a.shape[1], a.shape[2]))
    for i in range(a.shape[0]):
        if ((i % math.ceil(a.shape[0] / 10)) == 0):
            print("Done: ", i, "/", a.shape[0])
        for j in range(a.shape[1]):
            for k in range(a.shape[2]):
                if (a.ndim == 4):
                    for l in range(a.shape[3]):
                        shift_a[i, j, k, l] = a[i, j, k, l] + abs(min[l])
                else:
                    shift_a[i, j, k] = a[i, j, k] + abs(min)
    return shift_a


def nor_with_min_max(a, c, min, max):
    # if c > 0 then it applied the normalization of x: log(1+c*x)/log(1+c)
    print("Normalizing with given min max...")
    if (a.ndim == 4):
        norm_a = np.zeros((a.shape[0], a.shape[1], a.shape[2], a.shape[3]))
    elif (a.ndim == 1):
        norm_a = np.zeros((a.shape[0]))
    else:
        norm_a = np.zeros((a.shape[0], a.shape[1], a.shape[2]))
    for i in range(a.shape[0]):
        if ((i % math.ceil(a.shape[0] / 10)) == 0):
            print("Done: ", i, "/", a.shape[0])
        if (a.ndim == 1):
            norm_value = (a[i] - min) / (max - min)
            if (c > 0):
                norm_value = math.log(1 + c * norm_value) / math.log(1 + c)
            norm_a[i] = norm_value
            continue
        for j in range(a.shape[1]):
            for k in range(a.shape[2]):
                if (a.ndim == 4):
                    for l in range(a.shape[3]):
                        norm_value = (a[i, j, k, l] - min[l]) / (max[l] - min[l])
                        if (c > 0):
                            norm_value = math.log(1 + c * norm_value) / math.log(1 + c)
                        norm_a[i, j, k, l] = norm_value
                else:
                    norm_value = (a[i, j, k] - min) / (max - min)
                    if (c > 0):
                        norm_value = math.log(1 + c * norm_value) / math.log(1 + c)
                    norm_a[i, j, k] = norm_value
    return norm_a


def nor_a_ab(a, c, min, max):
    # c = 0: normalization MIN-MAX
    # c > 0: each value x is converted by applying the logarithic function new_x = log(1+x)
    print("Normalizing a with AB approach...")
    print("Computing min,max...")
    if (c > 0):
        a_norm = np.log(1 + c * a)
    else:
        a_norm = a
    minimum = math.log(1 + c * min)
    if (min == -1):
        minimum = np.amin(a_norm, axis=(0, 1, 2))
    maximum = math.log(1 + c * max)
    if (max == -1):
        maximum = np.amax(a_norm, axis=(0, 1, 2))
    for z_dim in range(a_norm.shape[3]):
        a_norm[:, :, :, z_dim] = (a_norm[:, :, :, z_dim] - minimum[z_dim]) / \
                                 (maximum[z_dim] - minimum[z_dim])
    #
    print("MIN: if (c>0) them log(1+c*min) else min: ", minimum)
    print("MAX: if (c>0) them log(1+c*max) else max: ", maximum)
    return a_norm


def nor_g(g, c):
    print("Normalizing g...")
    norm_g = np.zeros((g.shape[0], g.shape[1], g.shape[2]))
    min = 10000000000.0
    max = -10000000000.0
    print("Computing min,max...")
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            for k in range(g.shape[2]):
                if (g[i, j, k] > max):
                    max = g[i, j, k]
                if (g[i, j, k] < min):
                    min = g[i, j, k]
    print(min)
    print(max)
    for i in range(g.shape[0]):
        if ((i % math.ceil(g.shape[0] / 10)) == 0):
            print("Done: ", i, "/", g.shape[0])
        for j in range(g.shape[1]):
            for k in range(g.shape[2]):
                norm_value = (g[i, j, k] - min) / (max - min)
                if (c > 0):
                    norm_g[i, j, k] = math.log(1 + c * norm_value) / math.log(1 + c)
                else:
                    norm_g[i, j, k] = norm_value
    return norm_g


def nor_y_ab(y, c, min, max):
    # c = 0: normalization MIN-MAX
    # c > 0: each value x is converted by applying the logarithic function new_x = log(1+c*x)
    # min = -1: the minimum is computed
    # max = -1: the maximum is computed
    print("Normalizing y with AB approach...")
    minimum = 0.0
    maximum = 1.0
    if (c > 0):
        y_norm = np.log(1 + c * y)
    else:
        y_norm = y
    if (min == -1):
        minimum = np.amin(y_norm, axis=(0))
    else:
        if (c > 0):
            minimum = math.log(1 + c * min)
        else:
            minimum = min

    if (max == -1):
        maximum = np.amax(y_norm, axis=(0))
    else:
        if (c > 0):
            maximum = math.log(1 + c * max)
        else:
            maximum = max
    y_norm = (y_norm - minimum) / (maximum - minimum)
    #
    print("MIN: if (c>0) them log(1+c*min) else min: ", minimum)
    print("MAX: if (c>0) them log(1+c*max) else max: ", maximum)
    return y_norm


#
def nor_g_ab(hist, c, min, max):
    # c = 0: normalization MIN-MAX
    # c > 0: each value x is converted by applying the logarithic function new_x = log(1+c*x)
    # min = -1: the minimum is computed
    # max = -1: the maximum is computed
    if (c):
        hist = np.log(1 + c * hist)
    minimum = np.amin(hist, axis=(0, 1, 2)) if min == -1 else min
    maximum = np.amax(hist, axis=(0, 1, 2)) if max == -1 else max
    print("MIN: if (c>0) them log(1+c*min) else min: ", minimum)
    print("MAX: if (c>0) them log(1+c*max) else max: ", maximum)

    if len(hist.shape) == 3:
        return (hist - minimum) / (maximum - minimum)

    for z_dim in range(hist.shape[3]):
        hist[:, :, :, z_dim] = (hist[:, :, :, z_dim] - minimum[z_dim]) / (maximum[z_dim] - minimum[z_dim])
    return hist


def denorm_y_ab(y_nor, c, min, max):
    print("DeNoromalizing y..	.")
    min_log = math.log(1 + c * min)
    max_log = math.log(1 + c * max)
    delta = max_log - min_log
    y = np.exp(y_nor * delta - min_log)
    y = (y - 1) / c
    return y
