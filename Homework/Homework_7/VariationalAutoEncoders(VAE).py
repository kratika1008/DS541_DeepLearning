import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input, Flatten, Dense, Lambda, Reshape, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import backend as K

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train/255.
X_test = X_test/255.

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)
input_shape = (28, 28, 1)
latent_dim = 2

def vae_encoder(input_shape):
  e_input = Input(shape=input_shape, name='encoder_input')
  x = Conv2D(filters=16, kernel_size=3, padding='same', activation='relu')(e_input)
  x = BatchNormalization()(x)
  x = Conv2D(filters=32, kernel_size=3, strides=2, padding='same', activation='relu')(x)
  x = BatchNormalization()(x)
  conv_shape = K.int_shape(x)
  x = Flatten()(x)
  x = Dense(32, activation='relu')(x)
  x = BatchNormalization()(x)

  z_mu = Dense(latent_dim, name='latent_mu')(x)
  z_log_var = Dense(latent_dim, name='latent_variance')(x)

  return e_input, conv_shape, z_mu, z_log_var

e_input, conv_shape, z_mu, z_log_var = vae_encoder(input_shape)

def z_sampling(args):
    z_mu, z_log_var = args
    eps = K.random_normal(shape=(K.shape(z_mu)[0], latent_dim), mean=0., stddev=1.)
    return z_mu + K.exp(z_log_var) * eps

z = Lambda(z_sampling, output_shape=(latent_dim, ), name='z')([z_mu, z_log_var])

encoder = Model(e_input, [z_mu, z_log_var, z], name='encoder')

def vae_decoder(conv_shape):
  d_input = Input(shape=(latent_dim, ), name='decoder_input')
  x = Dense(conv_shape[1] * conv_shape[2] * conv_shape[3], activation='relu')(d_input)
  x = BatchNormalization()(x)
  x = Reshape((conv_shape[1], conv_shape[2], conv_shape[3]))(x)
  x = Conv2DTranspose(filters=32, kernel_size=3, padding='same', activation='relu')(x)
  x = BatchNormalization()(x)
  x = Conv2DTranspose(filters=16, kernel_size=3, strides=2, padding='same', activation='relu')(x)
  x = BatchNormalization()(x)
  x = Conv2DTranspose(filters=1, kernel_size=3, activation='sigmoid', padding='same', name='decoder_output')(x)

  return d_input, x

d_input, x = vae_decoder(conv_shape)
decoder = Model(d_input, x, name='decoder')



def vae_loss(x, x_decoded):
  flatten_x = K.flatten(x)
  flatten_x_decoded = K.flatten(x_decoded)
  rec_loss = binary_crossentropy(flatten_x, flatten_x_decoded) * 28 * 28
  kl_loss = 1 + z_log_var - K.square(z_mu) - K.exp(z_log_var)
  kl_loss = K.sum(kl_loss, axis=-1)
  kl_loss *= -0.8

  return K.mean(rec_loss + kl_loss)

vae_outputs = decoder(encoder(e_input)[2])
vae = Model(e_input, vae_outputs, name='vae')
vae.summary()

vae.compile(optimizer='adam', loss=vae_loss)

vae.fit(X_train, X_train, epochs=50, batch_size=16, validation_split=0.3)

X_test_sample = X_test[:16,::]
X_latent = encoder.predict(X_test_sample,batch_size=16)
for i in range(len(X_latent)):
  X_decoded = decoder.predict(X_latent[i])

#Display Original Mnist dataset images (first 16)
plt.figure(figsize=(20, 4))
for i in range(16):
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(X_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    #Display VAE decoded Mnist dataset images
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(X_decoded[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

# Vector of random points from Uniform normal distribution
random_points = np.random.rand(16,2)
# Decode random points to handwritten digit
encoded_img = decoder.predict(random_points)

plt.figure(figsize=(20, 4))
for i in range(16):
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(encoded_img[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)