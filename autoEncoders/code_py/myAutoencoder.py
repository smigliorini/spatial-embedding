#!/usr/bin/env python3
import math as mt
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
  def __init__(self, latent_dim, dimx, dimy, dimz, f1, f2):
    super(AutoencoderCNN_local, self).__init__()
    self.latent_dim = latent_dim
    self.dimx = dimx  
    self.dimy = dimy
    self.dimz = dimz
    self.encoder = tf.keras.Sequential([
      layers.Input(shape=(dimx, dimy, dimz)),
      layers.Conv2D(f1, (3, 3), activation='relu', padding='same', strides=2),
      # layers.MaxPooling2D(pool_size=(2, 2)),
      # layers.Conv2D(latent_dim*4, (3, 3), activation='relu', padding='same', strides=2),
      # prima era: latent_dim*2
      # layers.Conv2D(latent_dim*2, (3, 3), activation='relu', padding='same', strides=2),
      # prima era: latent_dim
      layers.Conv2D(f2, (3, 3), activation='relu', padding='same', strides=2),
      # layers.MaxPooling2D(pool_size=(2, 2)),
      # layers.Flatten(),
      # layers.Dense(int(dimx/4*dimy/4*latent_dim), activation='relu'),
      # layers.Dense(latent_dim, activation='relu'),
      layers.Conv2D(int(latent_dim/(dimx/8*dimy/8)), (3, 3), activation='relu', padding='same', strides=2),
      # layers.Flatten(),
      # layers.Dense(latent_dim, activation='relu')
    ])
    self.decoder = tf.keras.Sequential([
      # eliminare i due layer seguenti
      # layers.Dense(int(3*dimx/4*dimy/4), activation='relu'),
      # layers.Dense(int(dimx/4*dimy/4*latent_dim), activation='relu'),
      # layers.Dense(latent_dim), activation='relu'),
      # layers.Reshape((int(dimx/4), int(dimy/4), latent_dim)),
      # layers.Reshape((int(dimx/8), int(dimy/8), int(latent_dim/(16*16)))),
      # prima era latent_dim
      # layers.Conv2DTranspose(latent_dim*2, kernel_size=3, strides=2, activation='relu', padding='same'),
      # layers.Conv2DTranspose(f3, kernel_size=3, strides=2, activation='relu', padding='same'),
      # LAYER 0
      layers.Conv2DTranspose(int(latent_dim/(dimx/8*dimy/8)), kernel_size=3, strides=2, activation='relu', padding='same'),
      # 1 LAYER
      layers.Conv2DTranspose(f2, kernel_size=3, strides=2, activation='relu', padding='same'),
      # prima era latent_dim*2
      # layers.Conv2DTranspose(latent_dim*4, kernel_size=3, strides=2, activation='relu', padding='same'),
      # 2 LAYER
      layers.Conv2DTranspose(f1, kernel_size=3, strides=2, activation='relu', padding='same'),
      # 3 LAYER
      layers.Conv2D(dimz, kernel_size=3, activation='sigmoid', padding='same')
    ])
  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded
#
# CNN + dense mode
#
class AutoencoderCNNDense_local(Model):
  def __init__(self, latent_dim, dimx, dimy, dimz, f1, f2):
    super(AutoencoderCNNDense_local, self).__init__()
    self.latent_dim = latent_dim
    self.dimx = dimx
    self.dimy = dimy
    self.dimz = dimz
    self.encoder = tf.keras.Sequential([
      layers.Input(shape=(dimx, dimy, dimz)),
      layers.Conv2D(f1, (3, 3), activation='relu', padding='same', strides=2),
      # layers.MaxPooling2D(pool_size=(2, 2)),
      # layers.Conv2D(latent_dim*4, (3, 3), activation='relu', padding='same', strides=2),
      # prima era: latent_dim*2
      # layers.Conv2D(latent_dim*2, (3, 3), activation='relu', padding='same', strides=2),
      # prima era: latent_dim
      layers.Conv2D(f2, (3, 3), activation='relu', padding='same', strides=2),
      # layers.MaxPooling2D(pool_size=(2, 2)),
      # layers.Flatten(),
      # layers.Dense(int(dimx/4*dimy/4*latent_dim), activation='relu'),
      # layers.Dense(latent_dim, activation='relu'),
      # tolto 19/5 
      layers.Conv2D(int(latent_dim/(16*16)), (3, 3), activation='relu', padding='same', strides=2),
      layers.Flatten(),
      layers.Dense(latent_dim, activation='relu')
    ])
    self.decoder = tf.keras.Sequential([
      # eliminare i due layer seguenti
      # layers.Dense(int(3*dimx/4*dimy/4), activation='relu'),
      # tolto 19/5 layers.Dense(int(dimx/8)*int(dimy/8)*int(latent_dim/(16*16)), activation='relu'),
      # inserito al posto del layer sopra 19/5 il layer riga successiva
      layers.Dense(int(dimx/4)*int(dimy/4)*int(latent_dim/(16*16)), activation='relu'),
      # layers.Dense(latent_dim), activation='relu'),
      # layers.Reshape((int(dimx/4), int(dimy/4), latent_dim)),
      # tolto 19/5 layers.Reshape((int(dimx/8), int(dimy/8), int(latent_dim/(16*16)))),
      # inserito al posto del layer sopra 19/5 il layer riga successiva
      layers.Reshape((int(dimx/4), int(dimy/4), int(latent_dim/(16*16)))),
      # prima era latent_dim
      # layers.Conv2DTranspose(latent_dim*2, kernel_size=3, strides=2, activation='relu', padding='same'),
      # layers.Conv2DTranspose(f3, kernel_size=3, strides=2, activation='relu', padding='same'),
      # LAYER 0
      # tolto 19/5 layer sotto
      # layers.Conv2DTranspose(int(latent_dim/(16*16)), kernel_size=3, strides=2, activation='relu', padding='same'),
      # 1 LAYER
      layers.Conv2DTranspose(f2, kernel_size=3, strides=2, activation='relu', padding='same'),
      # prima era latent_dim*2
      # layers.Conv2DTranspose(latent_dim*4, kernel_size=3, strides=2, activation='relu', padding='same'),
      # 2 LAYER
      layers.Conv2DTranspose(f1, kernel_size=3, strides=2, activation='relu', padding='same'),
      # 3 LAYER
      layers.Conv2D(dimz, kernel_size=3, activation='sigmoid', padding='same')
    ])
  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

# AutoencoderCNNDense_local as sequential model ----------------------------- #
def AutoencoderCNNDense_local_noClass(f1, f2, latent_dim, dimx, dimy, dimz):
  # possible latent_dim: 3072, 2883, 2700, 2523, 2352, 2187, 2028, 1875, 1728, 1587, 1452, 1323, 1200, 1083, 972, 867, 
  #                      768, 675, 588, 507, 432, 363, 300, 243, 192, 147, 108, 75, 48 
  # INPUT
  input_hist = keras.Input(shape=(dimx,dimy,dimz))

  # "encoded" is the encoded representation of the input
  encoded = layers.Conv2D(f1, (3, 3), activation='relu', padding='same', strides=2)(input_hist) 
  encoded = layers.Conv2D(f2, (3, 3), activation='relu', padding='same', strides=2)(encoded)
  encoded = layers.Conv2D(int(latent_dim/(16*16)), (3, 3), activation='relu', padding='same', strides=2)(encoded)
  encoded = layers.Flatten()(encoded)
  encoded = layers.Dense(latent_dim, activation='relu')(encoded)
  # embeddings grid cell
  dim_x = mt.sqrt(latent_dim/3)
  dim_y = dim_x
  encoded = layers.Reshape((int(dim_x),int(dim_y),3))(encoded)

  # "decoded" is the lossy reconstruction of the input
  decoded = layers.Flatten()(encoded)
  decoded = layers.Dense(int(dimx/4)*int(dimy/4)*int(latent_dim/(16*16)), activation='relu')(decoded)
  decoded = layers.Reshape((int(dimx/4), int(dimy/4), int(latent_dim/(16*16))))(decoded)
  decoded = layers.Conv2DTranspose(f2, kernel_size=3, strides=2, activation='relu', padding='same')(decoded)
  decoded = layers.Conv2DTranspose(f1, kernel_size=3, strides=2, activation='relu', padding='same')(decoded)
  decoded = layers.Conv2D(dimz, kernel_size=3, activation='sigmoid', padding='same')(decoded)

  # AUTOENCODER model
  autoencoder = keras.Model(input_hist, decoded)
  autoencoder.summary()
  
  # ENCODER model
  encoder = keras.Model(input_hist, encoded) 
  encoder.summary()

  # DECODER model
  encoded_input = keras.Input(shape=(int(dim_x),int(dim_y),3))
  decoded_bis = autoencoder.layers[-6](encoded_input)
  decoded_bis = autoencoder.layers[-5](decoded_bis)
  decoded_bis = autoencoder.layers[-4](decoded_bis)
  decoded_bis = autoencoder.layers[-3](decoded_bis)
  decoded_bis = autoencoder.layers[-2](decoded_bis)
  decoded_bis = autoencoder.layers[-1](decoded_bis)
  decoder = keras.Model(encoded_input, decoded_bis)
  decoder.summary()

  return autoencoder, encoder, decoder
#
# DENSE model
#
class Autoencoder_local(Model):
  def __init__(self, f1, f2, latent_dim, dimx, dimy, dimz):
    super(Autoencoder_local, self).__init__()
    self.latent_dim = latent_dim
    self.dimx = dimx  
    self.dimy = dimy
    self.encoder = tf.keras.Sequential([
      layers.Flatten(),
      layers.Dense(f1, activation='relu'),
      layers.Dense(f2, activation='relu'),
      layers.Dense(latent_dim, activation='relu'),
    ])
    if (dimz == 1):
      self.decoder = tf.keras.Sequential([
	layers.Dense(f2, activation='relu'),
	layers.Dense(f1, activation='relu'),
        layers.Dense(dimx*dimy*dimz ,activation='sigmoid'),
        layers.Reshape((dimx, dimy)),
    ])
    else:
      self.decoder = tf.keras.Sequential([
	layers.Dense(f2, activation='relu'),
        layers.Dense(f1, activation='relu'),
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
    self.dimx = dimx  
    self.dimy = dimy
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
    self.dimx = dimx
    self.dimy = dimy
    self.encoder = tf.keras.Sequential([
      layers.Input(shape=(dimx, dimy, 1)),
      layers.Conv2D(latent_dim*4, (3, 3), activation='relu', padding='same', strides=2),
      layers.Conv2D(latent_dim*2, (3, 3), activation='relu', padding='same', strides=2),
      layers.Flatten(),
      layers.Dense(int(dimx/4*dimy/4*latent_dim), activation='relu'),
    ])
    self.decoder = tf.keras.Sequential([
      layers.Dense(int(dimx/4*dimy/4*latent_dim), activation='relu'),
      layers.Reshape((int(dimx/4), int(dimy/4), latent_dim)),
      layers.Conv2DTranspose(latent_dim*2, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.Conv2DTranspose(latent_dim*4, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same'),
    ])
  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded
#
