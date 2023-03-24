#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses
from keras.constraints import nonneg

#
# defining a new model for estimating join selectivity
#
# 2 INPUT models
#
class JN_2Input_DENSE2L_DENSE2L_DENSE2L(Model):
  def __init__(self, dimx, dimy, f1, f2 ,f3, f4, f5):
    # f1=16, f2=32 f3=64
    # f1=16, f2=32 f3=64
    super(JN_2Input_DENSE2L_DENSE2L_DENSE2L, self).__init__()
    self.dimx = dimx
    self.dimy = dimy
    self.f1 = f1
    self.f2 = f2
    self.f3 = f3
    self.f4 = f4
    self.f5 = f5
    self.hidden1 = keras.layers.Dense(f1, activation="relu")
    self.hidden2 = keras.layers.Dense(f2, activation="relu")
    self.hidden3 = keras.layers.Dense(f1, activation="relu")
    self.hidden4 = keras.layers.Dense(f2, activation="relu")
    self.hidden5 = keras.layers.Dense(f3, activation="relu")
    self.hidden6 = keras.layers.Dense(f4, activation="relu")
    self.hidden7 = keras.layers.Dense(f5, activation="relu")
    self.output_model = keras.layers.Dense(1, activation="linear")
    #, kernel_constraint=nonneg())
  def call(self, inputs):
    dataA, dataB = inputs
    flatA = keras.layers.Flatten()(dataA)
    flatB = keras.layers.Flatten()(dataB)
    h1 = self.hidden1(flatA)
    h2 = self.hidden2(h1)
    h3 = self.hidden3(flatB)
    h4 = self.hidden4(h3)
    concat = keras.layers.concatenate([h2, h4])
    h5 = self.hidden5(concat)
    h6 = self.hidden6(h5)
    h7 = self.hidden7(h6)
    #h7 = keras.layers.Dropout(0.2)(h7)
    out = self.output_model(h7)
    return out
#
class JN_2Input_CNN2L_CNN1L_DENSE2L(Model):
  def __init__(self, dimx, dimy, f1, f2 ,f3, f4):
    # f1=16, f2=32 f3=64
    # f1=16, f2=32 f3=64
    super(JN_2Input_CNN2L_CNN1L_DENSE2L, self).__init__()
    self.dimx = dimx
    self.dimy = dimy
    self.f1 = f1
    self.f2 = f2
    self.f3 = f3
    self.f4 = f4
    self.hidden1 = keras.layers.Conv2D(f1, (3, 3), activation='relu', padding='same', strides=2)
    #self.hidden1_b = keras.layers.BatchNormalization()
    #self.hidden1_mp = keras.layers.MaxPooling2D(pool_size=(2, 2))
    self.hidden2 = keras.layers.Conv2D(f2, (3, 3), activation='relu', padding='same', strides=2)
    #self.hidden2_b = keras.layers.BatchNormalization()
    #self.hidden2_mp = keras.layers.MaxPooling2D(pool_size=(2, 2))
    self.hidden3 = keras.layers.Conv2D(f1, (3, 3), activation='relu', padding='same', strides=2)
    #self.hidden3_b = keras.layers.BatchNormalization()
    #self.hidden3_mp = keras.layers.MaxPooling2D(pool_size=(2, 2))
    self.hidden4 = keras.layers.Dense(f3, activation="relu")
    self.hidden5 = keras.layers.Dense(f4, activation="relu")
    self.output_model = keras.layers.Dense(1, activation="linear")
    #, kernel_constraint=nonneg())
  def call(self, inputs):
    dataA, dataB = inputs
    x1 = self.hidden1(dataA)
    #x1 = self.hidden1_b(x1)
    #x1 = self.hidden1_mp(x1)
    x1 = self.hidden2(x1)
    #x1 = self.hidden2_b(x1)
    #x1 = self.hidden2_mp(x1)
    flatB = keras.layers.Flatten()(x1)
    x2 = self.hidden3(dataB)
    #x2 = self.hidden3_b(x2)
    #x2 = self.hidden3_mp(x2)
    flatA = keras.layers.Flatten()(x2)
    concat = keras.layers.concatenate([flatA, flatB])
    x = self.hidden4(concat)
    x = self.hidden5(x)
    x = keras.layers.Dropout(0.2)(x)
    out = self.output_model(x)
    return out
#
class JN_2Input_CNN2L_CNN1L_noBN_DENSE2L(Model):
  def __init__(self, dimx, dimy, f1, f2 ,f3, f4):
    # f1=16, f2=32 f3=64
    # f1=16, f2=32 f3=64
    super(JN_2Input_CNN2L_CNN1L_noBN_DENSE2L, self).__init__()
    self.dimx = dimx
    self.dimy = dimy
    self.f1 = f1
    self.f2 = f2
    self.f3 = f3
    self.f4 = f4
    self.hidden1 = keras.layers.Conv2D(f1, (3, 3), activation='relu', padding='same', strides=2)
    self.hidden2 = keras.layers.Conv2D(f2, (3, 3), activation='relu', padding='same', strides=2)
    self.hidden2_mp = keras.layers.MaxPooling2D(pool_size=(2, 2))
    self.hidden3 = keras.layers.Conv2D(f1, (3, 3), activation='relu', padding='same', strides=2)
    self.hidden3_mp = keras.layers.MaxPooling2D(pool_size=(2, 2))
    self.hidden4 = keras.layers.Dense(f3, activation="relu")
    self.hidden5 = keras.layers.Dense(f4, activation="relu")
    self.output_model = keras.layers.Dense(1, activation="linear")
    #, kernel_constraint=nonneg() )
  def call(self, inputs):
    dataA, dataB = inputs
    x1 = self.hidden1(dataA)
    x1 = self.hidden2(x1)
    x1 = self.hidden2_mp(x1)
    flatB = keras.layers.Flatten()(x1)
    x2 = self.hidden3(dataB)
    x2 = self.hidden3_mp(x2)
    flatA = keras.layers.Flatten()(x2)
    concat = keras.layers.concatenate([flatA, flatB])
    x = self.hidden4(concat)
    x = self.hidden5(x)
    x = keras.layers.Dropout(0.2)(x)
    out = self.output_model(x)
    return out
#
class JN_2Input_CNN2L_Conc_noBN_DENSE2L(Model):
  def __init__(self, dimx, dimy, f1, f2 ,f3, f4):
    # f1=16, f2=32 f3=64
    # f1=16, f2=32 f3=64
    super(JN_2Input_CNN2L_Conc_noBN_DENSE2L, self).__init__()
    self.dimx = dimx
    self.dimy = dimy
    self.f1 = f1
    self.f2 = f2
    self.f3 = f3
    self.f4 = f4
    self.hidden1 = keras.layers.Conv2D(f1, (3, 3), activation='relu', padding='same', strides=2)
    self.hidden2 = keras.layers.Conv2D(f2, (3, 3), activation='relu', padding='same', strides=2)
    self.hidden2_mp = keras.layers.MaxPooling2D(pool_size=(2, 2))
    self.hidden4 = keras.layers.Dense(f3, activation="relu")
    self.hidden5 = keras.layers.Dense(f4, activation="relu")
    self.output_model = keras.layers.Dense(1, activation="linear") 
    #, kernel_constraint=nonneg())
  def call(self, inputs):
    dataA, dataB = inputs
    x1 = self.hidden1(dataA)
    x1 = self.hidden2(x1)
    x1 = self.hidden2_mp(x1)
    flatB = keras.layers.Flatten()(x1)
    flatA = keras.layers.Flatten()(dataB)
    concat = keras.layers.concatenate([flatA, flatB])
    x = self.hidden4(concat)
    x = self.hidden5(x)
    x = keras.layers.Dropout(0.3)(x)
    out = self.output_model(x)
    return out
#
#
class JN_2Input_CNN2L_CNN2L_noBN_DENSE2L(Model):
  def __init__(self, dimx, dimy, f1, f2 ,f3, f4):
    # f1=16, f2=32 f3=64
    # f1=16, f2=32 f3=64
    super(JN_2Input_CNN2L_CNN2L_noBN_DENSE2L, self).__init__()
    self.dimx = dimx
    self.dimy = dimy
    self.f1 = f1
    self.f2 = f2
    self.f3 = f3
    self.f4 = f4
    self.hidden1 = keras.layers.Conv2D(f1, (3, 3), activation='relu', padding='same', strides=2)
    self.hidden2 = keras.layers.Conv2D(f2, (3, 3), activation='relu', padding='same', strides=2)
    self.hidden2_mp = keras.layers.MaxPooling2D(pool_size=(2, 2))
    self.hidden3 = keras.layers.Conv2D(f1, (3, 3), activation='relu', padding='same', strides=2)
    #self.hidden3_mp = keras.layers.MaxPooling2D(pool_size=(2, 2))
    self.hidden4 = keras.layers.Conv2D(f2, (3, 3), activation='relu', padding='same', strides=2)
    self.hidden4_mp = keras.layers.MaxPooling2D(pool_size=(2, 2))
    self.hidden5 = keras.layers.Dense(f3, activation="relu")
    self.hidden6 = keras.layers.Dense(f4, activation="relu")
    self.output_model = keras.layers.Dense(1, activation="linear")
  def call(self, inputs):
    dataA, dataB = inputs
    x1 = self.hidden1(dataB)
    x1 = self.hidden2(x1)
    x1 = self.hidden2_mp(x1)
    flatB = keras.layers.Flatten()(x1)
    x2 = self.hidden3(dataA)
    x2 = self.hidden4(x2)
    x2 = self.hidden4_mp(x2)
    flatA = keras.layers.Flatten()(x2)
    concat = keras.layers.concatenate([flatA, flatB])
    x = self.hidden5(concat)
    x = self.hidden6(x)
    x = keras.layers.Dropout(0.2)(x)
    out = self.output_model(x)
    return out
