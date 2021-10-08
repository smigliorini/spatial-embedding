#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses
#
# defining a new model for generating autoencoders for local histograms
#
# CNN model
#
class AutoencoderCNN_local(Model):
  def __init__(self, latent_dim, dimx, dimy, dimz):
    super(AutoencoderCNN_local, self).__init__()
    self.latent_dim = latent_dim
    self.encoder = tf.keras.Sequential([
      layers.Input(shape=(dimx, dimy, dimz)),
      layers.Conv2D(latent_dim*2, (3, 3), activation='relu', padding='same', strides=2),
      layers.Conv2D(latent_dim, (3, 3), activation='relu', padding='same', strides=2),
    ])
    self.decoder = tf.keras.Sequential([
      layers.Conv2DTranspose(latent_dim, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.Conv2DTranspose(latent_dim*2, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.Conv2D(dimz, kernel_size=(3, 3), activation='sigmoid', padding='same'),
    ])
  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded
#
# DENSE model
#
class Autoencoder_local(Model):
  def __init__(self, latent_dim, dimx, dimy, dimz):
    super(Autoencoder_local, self).__init__()
    self.latent_dim = latent_dim
    self.encoder = tf.keras.Sequential([
      layers.Flatten(),
      layers.Dense(latent_dim, activation='relu'),
    ])
    if (dimz == 1):
      self.decoder = tf.keras.Sequential([
        layers.Dense(dimx*dimy*dimz ,activation='sigmoid'),
        layers.Reshape((dimx, dimy)),
    ])
    else:
      self.decoder = tf.keras.Sequential([
        layers.Dense(dimx*dimy*dimz ,activation='sigmoid'),
        layers.Reshape((dimx, dimy, dimz)),
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded
#
# defining a new model for generating autoencoders for global histograms
#
# DENSE model
#
class Autoencoder_global(Model):
  def __init__(self, latent_dim, dimx, dimy):
    super(Autoencoder_global, self).__init__()
    self.latent_dim = latent_dim
    self.encoder = tf.keras.Sequential([
      layers.Flatten(),
      layers.Dense(latent_dim, activation='relu'),
    ])
    self.decoder = tf.keras.Sequential([
      layers.Dense(dimx*dimy ,activation='sigmoid'),
      layers.Reshape((dimx, dimy))
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded
# 
# CNN model
#
class AutoencoderCNN_global(Model):
  def __init__(self, latent_dim, dimx, dimy):
    super(AutoencoderCNN_global, self).__init__()
    self.latent_dim = latent_dim
    self.encoder = tf.keras.Sequential([
      layers.Input(shape=(dimx, dimy, 1)),
      layers.Conv2D(latent_dim*2, (3, 3), activation='relu', padding='same', strides=2),
      layers.Conv2D(latent_dim, (3, 3), activation='relu', padding='same', strides=2),
    ])
    self.decoder = tf.keras.Sequential([
      layers.Conv2DTranspose(latent_dim, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.Conv2DTranspose(latent_dim*2, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same'),
    ])
  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded
#
