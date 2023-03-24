# loading the models for local and global embedding
import time
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses
import generate_input_RQ as gt
import myModel_RQ as m
import myModel_JN as mj
from sklearn.model_selection import train_test_split
import plot as p
import balance_training_set as b
import csv

def extract (file_x, file_x1, file_y, file_ds, perc):
	x = np.load(file_x)
	x1 = np.load(file_x1)
	y = np.load(file_y)
	ds = np.load(file_ds)

	print("Splitting training and test set...")
	x, x_test, x1, x1_test, y, y_test, ds, ds_test = train_test_split(x, x1, y, ds, test_size=1-perc, random_state=42)
	np.save(file_x+str((1-perc)*100),x)
	np.save(file_x1+str((1-perc)*100),x1)
	np.save(file_y++str((1-perc)*100),y)
	np.save(fie_ds++str((1-perc)*100),ds)

def results_zero(ex_time, hh, a_y_test, a_y_pred, ds_test, x1_test, a_y_acc):
	np.savetxt("x1_test", x1_test, delimiter=',')
	with open("y_test_pred_distr", 'w') as f:
		csv.writer(f, delimiter=',').writerows(ds_test.tolist())
	for i in range(ex_time.shape[0]):
		print("execution_time[",i,"]: ",ex_time[i])
	for i in range(ex_time.shape[0]):
		last = len(hh[i].history["loss"])-1
		print("epochs: ",len(hh[i].history["loss"])," loss[",i,"]: ",hh[i].history["loss"][last]," - val_loss[",i,"]: ",hh[i].history["val_loss"][last])
	for i in range(a_y_test.shape[0]):
		y_test = a_y_test[i].reshape((a_y_test[i].shape[0]))
		y_pred = a_y_pred[i].reshape((a_y_test[i].shape[0]))
		y_test = np.reshape(y_test, (-1,1))
		y_pred = np.reshape(y_pred, (-1,1))
		y = np.concatenate((y_test,y_pred), axis=1)
		np.savetxt("y_test_pred_"+str(i)+"_"+str(a_y_acc[i]), y, delimiter=',')
		print("Accuracy[",i,"]:", a_y_acc[i])

def results(ex_dir, best_model, ex_time, hh, a_rma, a_mape, a_wmape, a_wmape_tot, a_mae_zero, mean_test, mean_pred, std_test, std_pred, freq, freq_zero, a_outliers, a_outliers_zero, a_y_test, a_y_pred, ds_test, x1_test):
	# create the directory for storing the experiment results
	os.mkdir(ex_dir)
	# move the best model to the experiment directory
	os.replace(best_model, ex_dir+"/"+best_model)

	for i in range(ex_time.shape[0]):
		print("execution_time[",i,"]: ",ex_time[i])
	for i in range(a_mape.shape[0]):
		last = len(hh[i].history["loss"])-1
		print("epochs: ",len(hh[i].history["loss"])," loss[",i,"]: ",hh[i].history["loss"][last]," - val_loss[",i,"]: ",hh[i].history["val_loss"][last])
	for i in range(a_mape.shape[0]):
		y_test = a_y_test[i].reshape((a_y_test[i].shape[0]))
		y_pred = a_y_pred[i].reshape((a_y_test[i].shape[0]))
		y_test = np.reshape(y_test, (-1,1))
		y_pred = np.reshape(y_pred, (-1,1))
		y = np.concatenate((y_test,y_pred), axis=1)

		# save results (actual, predicted) in a file
		np.savetxt(ex_dir+"/y_test_pred_"+str(i)+"_"+str(a_wmape_tot[i]), y, delimiter=',')

		y_test_gt0 = y_test[y_test>0]
		y_pred_gt0 = y_pred[y_test>0]
		mape = y_test_gt0 - y_pred_gt0
		mape = abs(mape)
		mape = mape/y_test_gt0
		mape = np.sort(mape)
		n = mape.shape[0]
		mape999 = mape[0:(int(n*0.999)-1)]
		mape999 = np.average(mape999)
		mape99 = mape[0:(int(n*0.99)-1)]
		mape99 = np.average(mape99)
		mape9 = mape[0:(int(n*0.9)-1)]
		mape9 = np.average(mape9)
		ratio = y_test_gt0/y_pred_gt0
		ratio = abs(ratio)
		inv = ratio<1
		ratio[inv] = 1/ratio[inv]
		ratio = np.sort(ratio)
		m = ratio.shape[0]
		rma999 = ratio[0:(int(m*0.999)-1)]
		rma999 = np.average(rma999)
		rma99 = ratio[0:(int(m*0.99)-1)]
		rma99 = np.average(rma99)
		rma9 = ratio[0:(int(m*0.9)-1)]
		rma9 = np.average(rma9)
		print("RMA[",i,"]:", a_rma[i], " RMA999,99,9:", rma999,rma99,rma9)
		print("MAPE[",i,"]:", a_mape[i], " freq[10^1,..,10^-5]: ",freq[i], " MAPE999,99,9:", mape999,mape99,mape9," WMAPE: ", a_wmape[i], " OutL: ", a_outliers[i])
	for i in range(a_mae_zero.shape[0]):
		print("MAE_0[",i,"]:", a_mae_zero[i], " freq[10^-1,..,10^-6]: ",freq_zero[i], " WMAPE_T: ", a_wmape_tot[i], " OutL_0: ", a_outliers_zero[i])
	for i in range(mean_test.shape[0]):
		print("mean_test[",i,"]:", mean_test[i], "(",std_test[i],") - mean_pred[",i,"]:", mean_pred[i], "(", std_pred[i],")")

def run_moretimes(zero, test_percentage, file_x1, file_x, file_y, file_ds, flag_zero, c, balance, times, start):
	# zero: 2 if the model is intended for classification, 
	#	1 if the model is intended for classifying ONLY zero and non zero cases, 
	#	0 otherwise (estimating values)
	# test_percentage: percentage of data points selected for test set
	# file_x1: 'XXX' means no file_x1 is present
	# file_x: file containing the data points describing the local/global embeddings
	# file_y: file containing the ground truth
	# file_ds: file containing the distributions of the datasets: 'XXX' means no file_x1 is present
	# flag_zero: 1 excludes zeros, 0 does not, a value S between 1 and 0 excludes all values that are > S
	# c: constant for normalization
	# balance: 0 means no balancing, 
	#          1 balancing by filling all the classes to reach a number equal to the class with the maximum number of elements
	#          2 balancing by filling all the classes to reach a number equal to the avarage number of elements in the classes
	hh = np.zeros((times,), dtype=dict)
	execution_time = np.zeros((times,), dtype=float)
	a_rma = np.zeros((times,), dtype=float)
	a_mape = np.zeros((times,), dtype=float)
	a_wmape = np.zeros((times,), dtype=float)
	a_wmape_tot = np.zeros((times,), dtype=float)
	a_outliers = np.zeros((times,), dtype=float)
	a_outliers_zero = np.zeros((times,), dtype=float)
	a_mae_zero = np.zeros((times,), dtype=float)
	mean_test = np.zeros((times,), dtype=np.double)
	mean_pred = np.zeros((times,), dtype=np.double)
	std_test = np.zeros((times,), dtype=np.double)
	std_pred = np.zeros((times,), dtype=np.double)
	a_freq = np.zeros((times,7), dtype=int)
	a_freq_zero = np.zeros((times,7), dtype=int)

	# loading x, x1, y and distr file
	# --------------------------------
	x = np.load(file_x)
	if (file_x1 != 'XXX'):
		x1 = np.load(file_x1)
	else:
		x1 = np.zeros((x.shape[0],1), dtype=np.double)
	
	y = np.load(file_y)
	if (file_ds != 'XXX'):
		ds = np.load(file_ds)
	else:
		ds = np.zeros((y.shape[0],2), dtype=float)

	if (flag_zero == 1):
		x = x[y>0.0]
		x1 = x1[y>0.0]
		ds = ds[y>0.0]
		y = y[y>0.0]
	elif (flag_zero > 0.0):
		print("Input x ",x.shape)
		print("Input x1 ",x1.shape)
		print("Input y ",y.shape)
		x = x[y<flag_zero]
		x1 = x1[y<flag_zero]
		ds = ds[y<flag_zero]
		y = y[y<flag_zero]
		print("Selected ",x.shape)

	dim = int(y.shape[0]*0.2)
	a_y_test = np.zeros((times,dim), dtype=np.double)
	a_y_pred = np.zeros((times,dim,1), dtype=np.double)
	a_y_test_class = np.zeros((times,dim), dtype=np.int8)
	a_y_pred_class = np.zeros((times,dim), dtype=np.int8)
	a_y_acc = np.zeros((times,), dtype=np.double)
	
	# normalizing y
	# ------------------------------------------
	if (zero > 0):
		y_nor = np.zeros((y.shape[0]), dtype=np.int8)
		if (zero == 1):
			for i in range(y.shape[0]):
				if (y[i] > 0):
					y_nor[i] = 1
			maximum = 1
		else:
			maximum = 10
	else:
		maximum = np.amax(y, axis=(0))
		minimum = np.amin(y, axis=(0))
		y_nor = gt.gh.nor_y_ab(y,c,minimum,maximum)
	freq = gt.gh.count_frequency_1(y_nor,10,1)
	p.plot_freq(freq)
	
	# augmenting to balance the training set
	# --------------------------------------------
	if (balance > 0):
		x_b, x1_b, y_b, ds_b = b.balance(x, x1, y_nor, ds, balance)	
		#freq_b = gt.gh.count_frequency_1(y_train_full,1)
		#p.plot_freq(freq_b)
	else:
		x_b = x
		x1_b = x1
		y_b = y_nor
		ds_b = ds

	# splitting test set and training test
	# ------------------------------------------
	print("Splitting training and test set...")
	x_train, x_test, x1_train, x1_test, y_train, y_test, ds_train, ds_test = train_test_split(x_b, x1_b, y_b, ds_b, test_size=test_percentage, random_state=42)

	# training models
	# --------------------------------------------	
	f1 = start
	for i in range(times):
		time0 = time.time()
		if (zero == 1):
			f2 = int(f1/2)
			f3 = int(f2/2)
			f4 = int(f3/2)
		else:
			f2 = int(f1/2)
			f3 = f2
			f4 = int(f1/4)
		print("Test: ",i+1,"/",times," Filters: ",f1,f2,f3,f4)
		if (zero > 0):
			mm, y_test_class, y_pred_class, h, x_test, x1_test, acc  = load_and_run_model(zero, test_percentage, x1_train , x_train, y_train, x1_test, x_test, y_test, maximum, flag_zero, c, f1, f2, f3, f4)
		else:
			mm, y_test_den, y_pred_den, h, x_loc_test, x_glo_test, rma, mape, wmape, mae_zero, freq, freq_zero, wmape_tot, outliers, outliers_zero = load_and_run_model(zero, test_percentage, x1_train , x_train, y_train, x1_test, x_test, y_test, maximum, flag_zero, c, f1, f2, f3, f4)
		execution_time[i] = time.time() - time0
		if (i == 0):
			best_model = mm
			if (zero > 0):
				best_acc = acc
			else:
				best_wmape = wmape_tot
		else:
			if (zero > 0):
				if (acc < best_acc):
					best_model = mm
					best_acc = acc
			else:
				if (wmape_tot < best_wmape):
					best_model = mm
					best_wmape = wmape_tot
		if (i == times-1):
			if (zero > 0):
				best_model_name = 'best_model_'+str(best_acc[0])
				best_model.save(best_model_name)
			else:
				best_model_name = 'best_model_'+str(best_wmape[0])
				best_model.save(best_model_name)
		hh[i] = h
		if (zero > 0):
			a_y_test_class[i] = y_test_class[0:dim]
			a_y_pred_class[i] = y_pred_class[0:dim]
			a_y_acc[i] = acc
		else:
			a_y_test[i] = y_test_den[0:dim]
			a_y_pred[i] = y_pred_den[0:dim]
			a_rma[i] = rma
			a_mape[i] = mape
			a_wmape[i] = wmape
			a_wmape_tot[i] = wmape_tot
			a_outliers[i] = outliers
			a_outliers_zero[i] = outliers_zero
			a_mae_zero[i] = mae_zero
			mean_test[i] = np.mean(y_test_den)
			mean_pred[i] = np.mean(y_pred_den)
			std_test[i] = np.std(y_test_den)
			std_pred[i] = np.std(y_pred_den)
			a_freq[i] = freq
			a_freq_zero[i] = freq_zero
			print("Test: ",i+1,"/",times,"DONE Filters: ",f1,f2,f3,f4, "MAPE, WMAPE, WMAPE_TOT, OUTLIERS",mape,wmape,wmape_tot,outliers)
		f1 = f1*2
	if (zero > 0):
		return best_model_name, ds_test, execution_time, hh, a_y_test_class, a_y_pred_class, a_y_acc, x1_test
	else:
		return best_model_name, ds_test, execution_time, hh, a_rma, a_mape, a_wmape, a_wmape_tot, a_mae_zero, mean_test, mean_pred, std_test, std_pred, a_freq, a_freq_zero, a_outliers, a_outliers_zero, a_y_test, a_y_pred, x1_test

# LOAD and TEST ---------------------------- #
def load_and_test_model(model_path, file_x1, file_x, file_y, file_ds):
	# Loading data points
	print("Loading data points...")
	x_test = np.load(file_x)
	x1_test = np.load(file_x1)
	y_test = np.load(file_y) 
	ds_test = np.load(file_ds)
	
	# Normalizing y
	mmin = 0.0
	mmax = 1.0
	c = 0
	y_test = gt.gh.nor_y_ab(y_test,c,mmin,mmax)

	freq = gt.gh.count_frequency_1(y_test,10,1)
	p.plot_freq(freq)
	
	# Loading model
	mymodel = keras.models.load_model(model_path)
	
	# Testing
	print("Model prediction...")
	print("x_test: ",x_test.shape)
	print("x1_test: ",x1_test.shape)
	y_pred = mymodel.predict([x_test, x1_test])
	print("y_test: ",y_test.shape)
	print("y_pred: ",y_pred.shape)	
	
	# Error 
	diff = abs(y_test - y_pred)
	print ("Norm MAE: ", np.mean(diff))
	rma0, mape0, wmape0, freq0, num0, mae_zero0, freq_zero0, num_zero0, wmapet0, out0, out_zero0 = mape_error_zero (y_test, y_pred)
	print ("Norm MAPE/WMAPE/Freq/Outliers/Tot_non_zero: ", mape0, wmape0, freq0, out0, num0)
	print ("Norm RMA non zero: ", rma0)
	print ("Norm MAE/WMAPETOT/Freq/Outliers/Tot_zero: ", mae_zero0, wmapet0, freq_zero0, out_zero0,  num_zero0)

	# ERROR evaluation DEnormalized VALUES --------------------------------
	#
	y_test_den = gt.gh.denorm_y_ab(y_test,c,mmin,mmax)
	y_pred_den = gt.gh.denorm_y_ab(y_pred,c,mmin,mmax)

	print ("Actual min/max: ", np.amin(y_test),"/", np.amax(y_test))
	print ("Prediction min/max: ", np.amin(y_pred),"/", np.amax(y_pred))
	diff_den = abs(y_test_den - y_pred_den)
	print ("MAE: ", np.mean(diff_den))
	rma, mape, wmape, freq, num, mae_zero, freq_zero, num_zero, wmape_tot, outliers, outliers_zero = mape_error_zero (y_test_den, y_pred_den)
	print ("MAPE/WMAPE/Freq/Outliers/Tot_non_zero: ", mape, wmape, freq, outliers, num)
	print ("RMA non zero: ", rma)
	print ("MAE/WMAPETOT/Freq/Outliers/Tot_zero: ", mae_zero, wmape_tot, freq_zero, outliers_zero, num_zero)
	return y_test_den, y_pred_den, rma, mape, wmape, mae_zero, freq, freq_zero, wmape_tot, outliers, outliers_zero
	

# LOAD AND RUN ----------------------------- #
def load_and_run_model(zero, test_percentage, x1, x, y, x1_test, x_test, y_test, y_maximum, flag_zero, c_norm, f1,f2,f3,f4):
	# x1: tensor containing the representation of the dataset MBR (4 values) 
	#     and the query rectangle (4 values)
	#     if np.amax(x1) = 0.0, then the global embedding and the query rectangle, represented 
	#     as an histogram of 32x32x1, are contained directly in the tensor x together with 
	#     the local embedding.
	print("Start load and run...")	
	# MODEL SELECTION ------------------
	#
	# RANGE QUERY ZERO DETECTION ------------------------------------------------------
	# mymodel = m.RQ_zero_2Input_CNN3L_DENSE2L(32,32,f1,f2,f3,f4)
	#
	# RANGE QUERY ---------------------------------------------------------
	# mymodel = m.RQ_sel_2Input_CNN2L_CNN1L_noBN_DENSE2L(32,32,f1,f2,f3,f4)
	# print("Model: RQ_sel_2Input_CNN2L_Conc_noBN_DENSE2L")
	# mymodel = m.RQ_sel_2Input_CNN2L_Conc_noBN_DENSE2L(32,32,f1,f2,f3,f4)
	# print("Model: RQ_sel_TL_2Input_CNN2L_Conc_noBN_DENSE2L")
	# autoencoder = "model/encoder_CNNDense_noClass_128-64_emb3072_synthetic"
	# mymodel = m.RQ_sel_TL_2Input_CNN2L_Conc_noBN_DENSE2L(autoencoder,128,128,f1,f2,f3,f4)
	# print("Model: RQ_sel_2Input_DENSE2L_DENSE3L")
	# mymodel = m.RQ_sel_2Input_DENSE2L_DENSE3L(32,32,f1,f2,f3,f3,f4)
	#
	# SPATIAL JOIN --------------------------------------------------------
	# mymodel = mj.JN_2Input_CNN2L_CNN1L_DENSE2L(32,32,f1,f2,f3,f4)
	# mymodel = mj.JN_2Input_CNN2L_CNN1L_noBN_DENSE2L(32,32,f1,f2,f3,f4)
	mymodel = mj.JN_2Input_CNN2L_Conc_noBN_DENSE2L(32,32,f1,f2,f3,f4)
	print("Model: JN_2Input_CNN2L_Conc_noBN_DENSE2L")
	# mymodel = mj.JN_2Input_DENSE2L_DENSE2L_DENSE2L(32,32,f1,f2,f3,f4,f4)
	# print("Model: JN_2Input_DENSE2L_DENSE2L_DENSE2L")

	# MODEL compile --------------------
	if (zero > 0):
		mymodel.compile(optimizer='adam', loss=losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
	else:
	 	mymodel.compile(optimizer='adam', loss=losses.MeanAbsoluteError())
	# 	mymodel.compile(optimizer='adam', loss=losses.MeanAbsolutePercentageError())
	#	mymodel.compile(optimizer='adam', loss=losses.MeanSquaredError())

	# SPLITTING Training and validation set -------------------------------
	#
	print("Splitting train and validation set...")
	if (np.amax(x1) == 0.0):
		print("Case RQ with all embeddings together: 32x32x3 (local) 32x32x2 (global) 32x32x1 (query rect)")
		X_train, X_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=42)
		X_loc_train = X_train[:,:,:,0:3]
		X_loc_valid = X_valid[:,:,:,0:3]
		X_glo_train = X_train[:,:,:,3:6]
		X_glo_valid = X_valid[:,:,:,3:6]
	else:
		print("Case RQ or JN with local embedding separated from global embedding")
		X_loc_train, X_loc_valid, X_glo_train, X_glo_valid, y_train, y_valid = train_test_split(x, x1, y, test_size=0.2, random_state=43) # it was 42

	# MODEL training ------------------------------------------------------
	#
	ep = 80
	# ep = 100
	print("Model training...")
	print("Epoch: ",ep)
	print("x_train: ",X_loc_train.shape)
	print("x1_train: ",X_glo_train.shape)
	print("y_train: ",y_train.shape)
 
	callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=6)

	# transfer learning
	# TO ACTIVATE TRANSFER LEARNING DECOMMENT THE FOLLOWING ROWS labelled with TL
	#for layer in mymodel.layers[0:7]: #TL
	#	layer.trainable = False #TL

	#mymodel.compile(optimizer='adam', loss=losses.MeanAbsoluteError()) #TL
	#h = mymodel.fit([X_loc_train, X_glo_train], y_train, epochs=10, batch_size=8, shuffle=True, callbacks=[callback], validation_data=([X_loc_valid, X_glo_valid], y_valid))
	#ep = ep - 10
	
	# autoe_layer.trainable = True
	#for layer in mymodel.layers[0:7]:
	#	layer.trainable = True
	#TL fino a qui
	mymodel.compile(optimizer='adam', loss=losses.MeanAbsoluteError())

	# normal learning
	history = mymodel.fit([X_loc_train, X_glo_train], y_train, epochs=ep, batch_size=8, shuffle=True, callbacks=[callback], validation_data=([X_loc_valid, X_glo_valid], y_valid))

	# MODEL summay --------------------------------------------------------
	#
	mymodel.summary()

	# MODEL testing -------------------------------------------------------
	#
	if (np.amax(x1) == 0.0):
		x1_test = x_test[:,:,:,3:6]
		x_test = x_test[:,:,:,0:3]
	
	# MODEL prediction ----------------------------------------------------
	#
	print("Model prediction...")
	print("x_test: ",x_test.shape)
	print("x1_test: ",x1_test.shape)
	y_pred = mymodel.predict([x_test, x1_test])
	print("y_test: ",y_test.shape)
	print("y_pred: ",y_pred.shape)
	# ERROR evaluation NORMALIZED VALUES ----------------------------------
	#
	if (zero > 0):
		y_pred_class = np.zeros((y_pred.shape[0]), dtype=np.int8)
		if (zero == 1):
			for i in range (y_pred.shape[0]):
				if (y_pred[i][0] > y_pred[i][1]):
					y_pred_class[i] = 0
				else:
					y_pred_class[i] = 1
			
			diff = y_test - y_pred_class
			num_ok = np.count_nonzero(diff==0)
			num_0for1 = np.count_nonzero(diff==-1)
			num_1for0 = np.count_nonzero(diff==1)
			accuracy = num_ok/len(diff)
			print ("ACCURACY (test set): ", accuracy, " 0for1: ", num_0for1, " 1for0: ", num_1for0 )
			return mymodel, y_test, y_pred_class, history, x_test, x1_test, accuracy
		else:
			for i in range(y_pred.shape[0]):
				class0 = 0
				max_acc = 0.0
				for j in range(y_pred[i]):
					if (y_pred[i,j] > max_acc):
						class0 = j
						max_acc = y_pred[i,j]
				y_pred_class[i] = class0
			diff = y_test - y_pred_class
			num_ok = np.count_nonzero(diff==0)
			accuracy = num_ok/len(diff)
			print ("ACCURACY (test set): ", accuracy )
			return mymodel, y_test, y_pred_class, history, x_test, x1_test, accuracy
	else:
		diff = abs(y_test - y_pred)
		print ("Norm MAE: ", np.mean(diff))
		print ("Actual min/max: ", np.amin(y_test),"/", np.amax(y_test))
		print ("Prediction min/max: ", np.amin(y_pred),"/", np.amax(y_pred))
		rma0, mape0, wmape0, freq0, num0, mae_zero0, freq_zero0, num_zero0, wmapet0, out0, out_zero0 = mape_error_zero (y_test, y_pred)
		print ("Norm MAPE/WMAPE/Freq/Outliers/Tot_non_zero: ", mape0, wmape0, freq0, out0, num0)
		print ("Norm RMA non zero: ", rma0)
		print ("Norm MAE/WMAPETOT/Freq/Outliers/Tot_zero: ", mae_zero0, wmapet0, freq_zero0, out_zero0,  num_zero0)

		# ERROR evaluation DEnormalized VALUES --------------------------------
		#
		y_test_den = gt.gh.denorm_y_ab(y_test,c_norm,0.0,y_maximum)
		y_pred_den = gt.gh.denorm_y_ab(y_pred,c_norm,0.0,y_maximum)
		
		diff_den = abs(y_test_den - y_pred_den)
		print ("Denor MAE: ", np.mean(diff_den))
		rma, mape, wmape, freq, num, mae_zero, freq_zero, num_zero, wmape_tot, outliers, outliers_zero = mape_error_zero (y_test_den, y_pred_den)
		print ("Denor MAPE/WMAPE/Outliers/Tot ONLY non_zero values: ", mape, wmape, outliers, num)
		print ("Denor RMA non zero: ", rma)
		print ("Denor Freq non zero predictions (relative error) [10, 1, 0.1, ..., 0.00001]:", freq)
		print ("Denor MAE/WMAPETOT/Outliers/Tot INCLUDING zero values: ", mae_zero, wmape_tot, outliers+outliers_zero, num+num_zero)
		print ("Denor Freq zero predictions (relative error) [10, 1, 0.1, ..., 0.00001]:", freq_zero)
		return mymodel, y_test_den, y_pred_den, history, x_test, x1_test, rma, mape, wmape, mae_zero, freq, freq_zero, wmape_tot, outliers, outliers_zero

def run_model():
	local_enc = keras.models.load_model('model/model_3072_CNNDense_newDatasets_SMLG')
	global_enc = keras.models.load_model('model/model_2048_CNNDense_newDatasets_SMLG_global_new')

	x, y = gt.gen_rq_input_from_file(local_enc, global_enc, 'rq/rq_newDatasets.csv', 'mbr/mbr_alldatasets.csv', 'rq/rq_result.csv', 'dataset-summaries.csv', 'histograms/new_datasets/', ';')

	#
	print("Inizilizing model DENSE local...")
	filter = 128
	model_rq = m.RQ_sel_DENSE(32,32,7,filter)
	model_rq.compile(optimizer='adam', loss=losses.MeanSquaredError())
	print("Splitting training and test set...")
	X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

	# training
	print("Training model...")
	history = model_rq.fit(X_train, y_train, batch_size=8, epochs=0, shuffle=True, validation_data=(X_test, y_test))
	return x, y, X_train, X_test, y_train, y_test, model_rq, history

def mape_error_zero (y, predict):
	delta_zero = 0.0
	zero = 0
	non_zero = 0
	outliers = 0
	outliers_zero = 0
	delta = 0.0
	delta_w = 0.0
	den_w = 0.0
	p = 0
	rma = 0.0
	freq_zero = np.zeros((7), dtype=int)
	freq = np.zeros((7), dtype=int)
	for i in range(y.shape[0]):
		val = predict[i]
		if (val < 0.0):
			val = 0.0
		if (y[i] == 0.0):
			zero += 1
			delta_zero += val
			if (val == 0.0):
				freq_zero[6] += 1
			elif (val < 0.000001):
				freq_zero[5] += 1
			elif (val < 0.00001):
				freq_zero[4] += 1
			elif (val < 0.0001):
				freq_zero[3] += 1
			elif (val < 0.001):
				freq_zero[2] += 1
			elif (val < 0.01):
				freq_zero[1] += 1
			elif (val < 0.1):
				freq_zero[0] += 1
			else:
				outliers_zero += 1
			if (zero < 10 and val > 0.1):
				print(y[i], val)
		else:
			non_zero += 1
			delta += abs(y[i] - val)/y[i]
			if (non_zero < 10 and (abs(y[i] - val)/y[i]) > 100):
				print("High relative error: ",abs(y[i] - val)/y[i])
			delta_w += abs(y[i] - val)
			a = abs(predict[i]/y[i])
			if (a < 1.0):
				a = 1/a
			rma += a
			den_w += y[i]
			if (abs(y[i] - predict[i])/y[i] < 0.00001):
				freq[6] += 1
			elif (abs(y[i] - predict[i])/y[i] < 0.0001):
				freq[5] += 1
			elif (abs(y[i] - predict[i])/y[i] < 0.001):
				freq[4] += 1
			elif (abs(y[i] - predict[i])/y[i] < 0.01):
				freq[3] += 1
			elif (abs(y[i] - predict[i])/y[i] < 0.1):
				freq[2] += 1
			elif (abs(y[i] - predict[i])/y[i] < 1):
				freq[1] += 1
			elif (abs(y[i] - predict[i])/y[i] < 10):
				freq[0] += 1
			else:
				outliers += 1
	
			if (p < 10 and abs(y[i] - predict[i])/y[i] > 50):
				print(y[i], predict[i])
				p += 1
	if (zero==0):
		zero = -1
	return rma/non_zero, delta/non_zero, delta_w/den_w , freq, non_zero, delta_zero/zero, freq_zero, zero, (delta_w+delta_zero)/den_w, outliers, outliers_zero
def zero_one (y):
	y_zero_one = np.zeros((y.shape[0]))
	for i in range(y.shape[0]):
		if (y[i] > 0.0):
			y_zero_one[i] = 1
	return y_zero_one
