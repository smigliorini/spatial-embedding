#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses
#
# defining a new model for generating autoencoders for local histograms
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
