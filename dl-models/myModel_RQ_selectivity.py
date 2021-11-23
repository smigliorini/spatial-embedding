#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses


#
# defining a new model for estimating range query selectivity
#
# CNN model
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
