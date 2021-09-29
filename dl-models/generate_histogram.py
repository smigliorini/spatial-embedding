#!/usr/bin/env python3
import os
from os import listdir, mkdir, path, sep
from os.path import isfile, join
import numpy as np
import random as rd
import math
import csv
def count_frequency(a,dimx,dimy):
	freq = np.zeros(11)
	for i in range(a.shape[0]):
		for j in range(dimx):
			for k in range(dimy):
				if (a[i,j,k] == 0):
					index = 0
				else:
					index = math.ceil(a[i,j,k]*10)
				if (index >= 11):
					index = 10
				freq[index] += 1
	return freq
def count_frequency_std(a,dimx,dimy):
	freq = np.zeros(22)
	for i in range(a.shape[0]):
		for j in range(dimx):
			for k in range(dimy):
				index = math.floor(a[i,j,k]*10) 
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
def gen_hist_from_file(dimx,dimy,dimz,file):
	h0 = np.zeros((dimx,dimy,dimz))
	with open(file, mode='r') as csv_file:
		csv_reader = csv.DictReader(csv_file)
		line_count = 0
		for row in csv_reader:
			if line_count == 0:
				print(f'Column names are: {", ".join(row)}')
				line_count += 1
			#DEBUG print(f'\t{row["i0"]},{row["i1"]}: {row["num_features"]}, {row["size"]}, {row["num_points"]}, {row["avg_area"]}, {row["avg_side_length_0"]}, {row["avg_side_length_1"]}.')
			x = int(row["i0"])
			y = int(row["i1"])
			if (dimz >= 1):
				h0[x,y,0] = int(row["num_features"])
			if (dimz >= 2):
				h0[x,y,1] = int(row["size"])
			if (dimz >= 3):
				h0[x,y,2] = int(row["num_points"])
			if (dimz >= 4):
				h0[x,y,3] = float(row["avg_area"])
			if (dimz >= 5):
				h0[x,y,4] = float(row["avg_side_length_0"])
			if (dimz >= 6):	
				h0[x,y,5] = float(row["avg_side_length_1"])
			line_count += 1
		print(f'Processed {line_count} lines.')
	return h0
def gen_input_from_file(dimx,dimy,dimz,path):
	files = get_files_path(path)
	print('Found {0} files'.format(len(files)))
	hh = np.zeros((len(files),dimx,dimy,dimz))
	count = 0
	# for each file, generate one histogram
	for ff in files:
		print('Processing file {0} ...'.format(ff))
		h0 = gen_hist_from_file(dimx,dimy,dimz,ff)
		hh[count] = h0
		count += 1
	return hh	
def cell_mod(dimz,card,delta,lenx,dx,leny,dy,area,dia,pnt,deltapnt,size,deltasize):
	if (dimz == 1):
		cloc = 0.0
	else:
		cloc = np.zeros(dimz)
	if (dimz > 0):
		if (dimz == 1):
			cloc = card + rd.randint(-delta,delta)
		else:
			cloc[0] = card + rd.randint(-delta,delta)
	if (dimz > 1):
                cloc[1] = pnt + rd.randint(-deltapnt,deltapnt)
	if (dimz > 2):
		cloc[2] = size + rd.randint(-deltasize,deltasize)
	if (dimz > 3):
		cloc[3] = 0.000001 + lenx + rd.randint(-1,1)*rd.random()*dx
	if (dimz > 4):
		cloc[4] = 0.000001 + leny + rd.randint(-1,1)*rd.random()*dy
	if (dimz > 5):
		cloc[5] = 0.0000001 + area + rd.randint(-1,1)*rd.random()*dx*dy
	return cloc
def generate(num,dimx,dimy,dimz,type):
	debug = 0
	if (dimz > 6):
		return "Max value for dimz is 6"
	if (dimz == 1):
		a = np.zeros((num,dimx,dimy))
	else:
		a = np.zeros((num,dimx,dimy,dimz))
	for i in range(num):
		if ((i % math.ceil(num/10)) == 0):
			print("Done: ",i,"/",num)
		card = rd.randint(10,1000)
		delta = math.ceil(0.1*card)
		lenx = rd.random()*0.2
		dx = 0.1*lenx
		leny = rd.random()*0.2
		dy = 0.1*leny
		area = lenx*leny
		dia = math.ceil(dimx/5)*rd.random()
		pnt = rd.randint(50,100000)
		deltapnt = math.ceil(0.1*pnt)
		size = rd.randint(100,10000)
		deltasize = math.ceil(0.1*size)
		vert = math.ceil(dimy*rd.random())
		diav = math.ceil(dimy/5)*rd.random()
		horiz = math.ceil(dimx*rd.random())
		diah = math.ceil(dimx/5)*rd.random()
		for j in range(dimx):
			for k in range(dimy):
				if (type == 0): # UNIFORM DISTRIBUTION
					a[i,j,k] = cell_mod(dimz,card,delta,lenx,dx,leny,dy,area,dia,pnt,deltapnt,size,deltasize)
				if (type == 1 and abs(j-k) < dia): # DIAGONAL DISTRIBUTION
					a[i,j,k] = cell_mod(dimz,card,delta,lenx,dx,leny,dy,area,dia,pnt,deltapnt,size,deltasize)
				if (type == 2 and (abs(k-vert) < diaiv)): # VERTICAL DISTR
					a[i,j,k] = cell_mod(dimz,card,delta,lenx,dx,leny,dy,area,dia,pnt,deltapnt,size,deltasize)
				if (type == 3 and (abs(j-horiz) < diah)): # HORIZONTAL DISTR
					a[i,j,k] = cell_mod(dimz,card,delta,lenx,dx,leny,dy,area,dia,pnt,deltapnt,size,deltasize)
				if (type == 4 and (abs(j-horiz) < diah) and (abs(k-vert) < diav)): # GAUSSIAN DISTR
					a[i,j,k] = cell_mod1(dimz,card,delta,lenx,dx,leny,dy,area,dia,pnt,deltapnt,size,deltasize)
				if (type == 5): # MIX
					if (i % 10 in [0,2,6,8]) or \
					   (i % 10 in [1,4] and (abs(j-k) < dia)) or \
					   (i % 10 == 3 and (abs(k-vert) < diav)) or \
					   (i % 10 == 5 and (abs(j-horiz) < diah)) or \
					   (i % 10 in [7,9] and (abs(j-horiz) < diah) and (abs(k-vert) < diav)):
						a[i,j,k] = cell_mod(dimz,card,delta,lenx,dx,leny,dy,area,dia,pnt,deltapnt,size,deltasize)
	if (debug == 1):
		for i in range(num):
			count = 0
			count_zero = 0
			for j in range(dimx):
				for k in range(dimy):
					count = count + 1
					if a[i,j,k,0] == 0.0:
						count_zero = count_zero + 1
			print("Riga: ",i," -> ",count_zero,"/",count)

	print("End generation of local histograms")
	print("Generating global histograms...")
	g = np.zeros((num,dimx,dimy))	
	for i in range(num):
		j = rd.randint(0,math.ceil(dimx*2/3))
		k = rd.randint(0,math.ceil(dimy*2/3))
		d_j = rd.randint(math.ceil(dimx/6),math.ceil(dimx/3))
		d_k = rd.randint(math.ceil(dimy/6),math.ceil(dimy/3))
		for jbis in range(d_j):
			for kbis in range(d_k):
				if (j+jbis < dimx and k+kbis < dimy):
					g[i,j+jbis,k+kbis] = rd.randint(0,8000)

	print("Generating random range query selectivities...")
	b = np.zeros(num)
	for i in range(num):
		if (rd.random() < 0.8):
			b[i] = rd.randint(0,1200)

	print("Generating range queries...")
	r = np.zeros((num,dimx,dimy))
	for i in range(num):
		j = rd.randint(0,dimx-1)
		k = rd.randint(0,dimy-1)
		d_j = rd.randint(1,math.ceil(dimx/25))
		d_k = rd.randint(1,math.ceil(dimy/25))
		for jbis in range(d_j):
			for kbis in range(d_k):
				if (j+jbis < dimx and k+kbis < dimy):
					r[i,j+jbis,k+kbis] = 1.0
	print("Returning a, g, b and r")
	return a, g, b, r
def prr(a):
	for i in range(a.shape[0]):
		for j in range(a.shape[1]):
			v = int(a[i,j]*10) % 10
			print(v, end='')
		print("#")
def prr_delta10(a,da):
	for i in range(a.shape[0]):
		for j in range(a.shape[1]):
			v = int(abs(a[i,j]-da[i,j])*10) % 10
			print(v, end='')
		print("#")	
def prr_delta100(a,da):
	avg = 0
	for i in range(a.shape[0]):
		avg = 0
		for j in range(a.shape[1]):
			v = int(abs(a[i,j]-da[i,j])*100) % 10
			avg = avg + v
			print(v, end='')
		print("#",int(avg)/a.shape[1])
def prr1(a,s1,s2):
	for i in range(a.shape[0]):
		for j in range(a.shape[1]):
			if (a[i,j] <= s1):
				print("0", end='')
			else:
				if (a[i,j] <= s2):
					print("-", end='')
				else:
					print("1", end='')
		print("#")
def prr_a(a,f,s1,s2):
	for i in range(a.shape[0]):
		for j in range(a.shape[1]):
			if (a[i,j,f] <= s1):
				print("0", end='')
			else:
				if (a[i,j,f] <= s2):
					print("-", end='')
				else:
					print("1", end='')
		print("#")
def std_a(a,zero):
	if (zero == 0): # zero=0 includes the zeros in the average
		print("Standardizing 'a' including zeros...")
	else: # zero=1 excludes the zeros in the average
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
						if (a[i,j,k,l] == 0):
							count_zero[l] += 1
						else:
							avg[l] = avg[l] + a[i,j,k,l]
				else:
					if (a[i,j,k] == 0):
						count_zero += 1
					else:
						avg = avg + a[i,j,k]
	
	if (a.ndim == 4):
		for l in range(a.shape[3]):
			den[l] = a.shape[0]*a.shape[1]*a.shape[2]
	else:
		den = a.shape[0]*a.shape[1]*a.shape[2]
	if (zero == 1):
		if (a.ndim == 4):
			for l in range(a.shape[3]):
				den[l] = den[l] - count_zero[l]
		else:
			den = den -count_zero
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
						if (zero == 0 or a[i,j,k,l] > 0):
							for l in range(a.shape[3]):
								mse[l] = mse[l] + (avg[l] - a[i,j,k,l])**2
					else:
						if (zero == 0 or a[i,j,k] > 0):
							mse = mse + (avg - a[i,j,k])**2
	if (a.ndim == 4):
		for l in range(a.shape[3]):
			mse[l] = pow(mse[l]/den[l],0.5)
	else:
		mse = pow(mse/den,0.5)

	if (a.ndim == 4):
		scal_a = np.zeros((a.shape[0],a.shape[1],a.shape[2],a.shape[3]))
	else:
		scal_a = np.zeros((a.shape[0],a.shape[1],a.shape[2]))

	for i in range(a.shape[0]):
		for j in range(a.shape[1]):
			for k in range(a.shape[2]):
				if (a.ndim == 4):
					for l in range(a.shape[3]):
						if (zero == 0 or a[i,j,k,l] > 0):
							scal_a[i,j,k,l] = (a[i,j,k,l] - avg[l])/mse[l]
				else:
					if (zero == 0 or a[i,j,k] > 0):
						scal_a[i,j,k] = (a[i,j,k] - avg)/mse  
	print("avg and mse of the local histograms")
	print(avg, mse)
	return scal_a
def std_g(g):
	print("Standadizing g...")
	scal_g = np.zeros((g.shape[0],g.shape[1],g.shape[2]))
	avg = 0
	mse = 0
	for i in range(g.shape[0]):
		for j in range(g.shape[1]):
			for k in range(g.shape[2]):
				avg = avg + g[i,j,k]
	den = g.shape[0]*g.shape[1]*g.shape[2]
	avg = avg/den
	for i in range(g.shape[0]):
		for j in range(g.shape[1]):
			for k in range(g.shape[2]):
				mse = mse + (avg - g[i,j,k])**2
	mse = pow(mse/den, 0.5)

	for i in range(g.shape[0]):
		for j in range(g.shape[1]):
			for k in range(g.shape[2]):
				scal_g[i,j,k] = (g[i,j,k] - avg)/mse 
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
	avg = avg/b.shape[0]
	for i in range(b.shape[0]):
		mse = mse + (avg - b[i])**2
	mse = pow(mse/b.shape[0], 0.5)

	for i in range(b.shape[0]):
		scal_b[i] = (b[i] - avg)/mse
	
	print("avg and mse of the selectivity of range queries")
	print(avg, mse)

	return scal_b
def shift_pos(a):
	print("Shifting a...")
	if (a.ndim == 4):
		min = np.ones(a.shape[3])*10000000000
	else:
		min = 10000000000
	print("Computing min...")
	for i in range(a.shape[0]):
		for j in range(a.shape[1]):
			for k in range(a.shape[2]):
				if (a.ndim == 4):
					for l in range(a.shape[3]):
						if (a[i,j,k,l]<min[l]):
							min[l] = a[i,j,k,l]
				else:
					if (a[i,j,k]<min):
						min = a[i,j,k]
	print("min: ",min)
	if (a.ndim == 4):
		shift_a = np.zeros((a.shape[0],a.shape[1],a.shape[2],a.shape[3]))
	else:
		shift_a = np.zeros((a.shape[0],a.shape[1],a.shape[2]))
	for i in range(a.shape[0]):
		if ((i % math.ceil(a.shape[0]/10)) == 0):
			print("Done: ",i,"/",a.shape[0])
		for j in range(a.shape[1]):
			for k in range(a.shape[2]):
				if (a.ndim == 4):
					for l in range(a.shape[3]):
						shift_a[i,j,k,l] = a[i,j,k,l] + abs(min[l])
				else:
					shift_a[i,j,k] = a[i,j,k] + abs(min)
	return shift_a
def nor_a(a):
	print("Normalizing a...")
	if (a.ndim == 4):
		min = np.ones(a.shape[3])*10000000000
		max = np.ones(a.shape[3])*(-10000000000)
	else:
		min = 10000000000
		max = (-10000000000)
	print("Computing min,max...")
	for i in range(a.shape[0]):
		for j in range(a.shape[1]):
			for k in range(a.shape[2]):
				if (a.ndim == 4):
					for l in range(a.shape[3]):
						if (a[i,j,k,l]<min[l]):
							min[l] = a[i,j,k,l]
						if (a[i,j,k,l]>max[l]):
							max[l] = a[i,j,k,l]
				else:
					if (a[i,j,k]<min):
						min = a[i,j,k]
					if (a[i,j,k]>max):
						max = a[i,j,k]
	#Debug
	print(min)
	print(max)
	if (a.ndim == 4):
		norm_a = np.zeros((a.shape[0],a.shape[1],a.shape[2],a.shape[3]))
	else:
		norm_a = np.zeros((a.shape[0],a.shape[1],a.shape[2]))
	for i in range(a.shape[0]):
		if ((i % math.ceil(a.shape[0]/10)) == 0):
                        print("Done: ",i,"/",a.shape[0])
		for j in range(a.shape[1]):
			for k in range(a.shape[2]):
				if (a.ndim == 4):
					for l in range(a.shape[3]):
						norm_a[i,j,k,l] = (a[i,j,k,l] - min[l])/(max[l]-min[l])
				else:
					norm_a[i,j,k] = (a[i,j,k] - min)/(max - min)
	return norm_a
def nor_g(g):
	print("Normalizing g...")
	norm_g = np.zeros((g.shape[0],g.shape[1],g.shape[2]))
	min = 10000000000.0
	max = -10000000000.0
	print("Computing min,max...")
	for i in range(g.shape[0]):
		for j in range(g.shape[1]):
			for k in range(g.shape[2]):
				if (g[i,j,k]>max):
					max = g[i,j,k]
				if (g[i,j,k]<min):
					min = g[i,j,k]
	for i in range(g.shape[0]):
		if ((i % math.ceil(g.shape[0]/10)) == 0):
                        print("Done: ",i,"/",g.shape[0])
		for j in range(g.shape[1]):
			for k in range(g.shape[2]):
				norm_g[i,j,k] = (g[i,j,k] - min)/(max - min)
	return norm_g
def nor_b(b):
	print("Normalizing b...")
	norm_b = np.zeros(b.shape[0])
	min = 10000000000
	max = -10000000000
	print("Computing min,max...")
	for i in range(b.shape[0]):
		if (b[i]>max):
			max = b[i]
		if (b[i]<min):
			min = b[i]
	print("Modifying values...")
	for i in range(b.shape[0]):
		norm_b[i] = (b[i] - min)/(max - min)
	
	return norm_b
