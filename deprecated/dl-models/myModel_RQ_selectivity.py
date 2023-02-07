#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses


#
# defining a new model for estimating range query selectivity
#
# 2 INPUT models
#
class RQ_sel_2Input_DENSE3L_DENSE2L(Model):
  def __init__(self, dimx, dimy, f1, f2 ,f3, f4, f5):
    # f1=16, f2=32 f3=64
    # f1=16, f2=32 f3=64
    super(RQ_sel_2Input_DENSE3L_DENSE2L, self).__init__()
    self.dimx = dimx
    self.dimy = dimy
    self.f1 = f1
    self.f2 = f2
    self.f3 = f3
    self.f4 = f4
    self.f5 = f5
    self.hidden1 = keras.layers.Dense(f1, activation="relu")
    self.hidden2 = keras.layers.Dense(f2, activation="relu")
    self.hidden3 = keras.layers.Dense(f3, activation="relu")
    self.hidden4 = keras.layers.Dense(f4, activation="relu")
    self.hidden5 = keras.layers.Dense(f5, activation="relu")
    self.output_model = keras.layers.Dense(1, activation="linear")
  def call(self, inputs):
    dataA, dataB = inputs
    flatA = keras.layers.Flatten()(dataA)
    flatB = keras.layers.Flatten()(dataB)
    h1 = self.hidden1(flatB)
    h2 = self.hidden2(h1)
    h3 = self.hidden3(h2)
    concat = keras.layers.concatenate([flatA, h2])
    h4 = self.hidden4(concat)
    h5 = self.hidden5(h4)
    #h4 = keras.layers.Dropout(0.2)(h3)
    out = self.output_model(h5)
    return out
#
class RQ_sel_2Input_CNN2L_CNN1L_DENSE2L(Model):
  def __init__(self, dimx, dimy, f1, f2 ,f3, f4):
    # f1=16, f2=32 f3=64
    # f1=16, f2=32 f3=64
    super(RQ_sel_2Input_CNN2L_CNN1L_DENSE2L, self).__init__()
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
  def call(self, inputs):
    dataA, dataB = inputs
    x1 = self.hidden1(dataB)
    #x1 = self.hidden1_b(x1)
    #x1 = self.hidden1_mp(x1)
    x1 = self.hidden2(x1)
    #x1 = self.hidden2_b(x1)
    #x1 = self.hidden2_mp(x1)
    flatB = keras.layers.Flatten()(x1)
    x2 = self.hidden3(dataA)
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
class RQ_sel_2Input_CNN2L_CNN1L_noBN_DENSE2L(Model):
  def __init__(self, dimx, dimy, f1, f2 ,f3, f4):
    # f1=16, f2=32 f3=64
    # f1=16, f2=32 f3=64
    super(RQ_sel_2Input_CNN2L_CNN1L_noBN_DENSE2L, self).__init__()
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
  def call(self, inputs):
    dataA, dataB = inputs
    x1 = self.hidden1(dataB)
    x1 = self.hidden2(x1)
    x1 = self.hidden2_mp(x1)
    flatB = keras.layers.Flatten()(x1)
    x2 = self.hidden3(dataA)
    x2 = self.hidden3_mp(x2)
    flatA = keras.layers.Flatten()(x2)
    concat = keras.layers.concatenate([flatA, flatB])
    x = self.hidden4(concat)
    x = self.hidden5(x)
    x = keras.layers.Dropout(0.2)(x)
    out = self.output_model(x)
    return out
#
#
class RQ_sel_2Input_CNN2L_CNN2L_noBN_DENSE2L(Model):
  def __init__(self, dimx, dimy, f1, f2 ,f3, f4):
    # f1=16, f2=32 f3=64
    # f1=16, f2=32 f3=64
    super(RQ_sel_2Input_CNN2L_CNN2L_noBN_DENSE2L, self).__init__()
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



#
# ONE INPUT models
# CNN models
#
class RQ_sel_CNN_3L_DENSE(Model):
    def __init__(self, dimx, dimy, dimz, f1, f2, f3):
        # f1=16, f2=32 f3=64
        # f1=16, f2=32 f3=64
        super(RQ_sel_CNN_3L_DENSE, self).__init__()
        self.dimx = dimx
        self.dimy = dimy
        self.dimz = dimz
        self.f1 = f1
        self.f2 = f2
        self.f3 = f3
        self.predictor = tf.keras.Sequential([
            # INPUT layer
            layers.Input(shape=(dimx, dimy, dimz)),
            # First CNN => BN => POOL layer
            layers.Conv2D(f1, (3, 3), activation='relu', padding='same', strides=2),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            # Second CNN => BN => POOL layer
            layers.Conv2D(f2, (3, 3), activation='relu', padding='same', strides=2),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            # Third CNN => BN => POOL layer
            layers.Conv2D(f3, (3, 3), activation='relu', padding='same', strides=2),
            layers.BatchNormalization(),
            # layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(f1, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(1, activation="linear")
        ])

    def call(self, x):
        sel = self.predictor(x)
        return sel


class RQ_sel_CNN_2L_DENSE(Model):
    def __init__(self, dimx, dimy, dimz, f1, f2):
        # f1=16 f2=32
        super(RQ_sel_CNN_2L_DENSE, self).__init__()
        self.dimx = dimx
        self.dimy = dimy
        self.dimz = dimz
        self.f1 = f1
        self.f2 = f2
        self.predictor = tf.keras.Sequential([
            # INPUT layer
            layers.Input(shape=(dimx, dimy, dimz)),
            # First CNN => BN => POOL layer
            layers.Conv2D(f1, (3, 3), activation='relu', padding='same', strides=2),
            layers.BatchNormalization(),
            # deleted layers.MaxPooling2D(pool_size=(2, 2)),
            # Second CNN => BN => POOL layer
            layers.Conv2D(f2, (3, 3), activation='relu', padding='same', strides=2),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(f1, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid")
        ])

    def call(self, x):
        sel = self.predictor(x)
        return sel


class RQ_sel_CNN_1L_DENSE(Model):
    def __init__(self, dimx, dimy, dimz, f1, f2):
        # f1=16 f2=32
        super(RQ_sel_CNN_1L_DENSE, self).__init__()
        self.dimx = dimx
        self.dimy = dimy
        self.dimz = dimz
        self.f1 = f1
        self.f2 = f2
        self.predictor = tf.keras.Sequential([
            # INPUT layer
            layers.Input(shape=(dimx, dimy, dimz)),
            # First CNN => BN => POOL layer
            layers.Conv2D(f2, (3, 3), activation='relu', padding='same', strides=2),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(f1, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            # layers.Dense(1, activation="linear"),
            layers.Dense(1, activation="relu")
        ])

    def call(self, x):
        sel = self.predictor(x)
        return sel


class RQ_sel_DENSE(Model):
    def __init__(self, dimx: int, dimy: int, dimz: int, units: int):
        # f1=16
        super(RQ_sel_DENSE, self).__init__()
        self.dimx = dimx
        self.dimy = dimy
        self.dimz = dimz
        self.units = units
        self.predictor = tf.keras.Sequential([
            # INPUT layer
            layers.Input(shape=(dimx, dimy, dimz)),
            layers.Flatten(),
            layers.Dense(units, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(1, activation="linear")
        ])

    def call(self, x):
        sel = self.predictor(x)
        return sel
