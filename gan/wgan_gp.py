from __future__ import print_function, division

import json

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers
from einops.layers.tensorflow import Rearrange
import os


class WGAN_GP(keras.Model):
    def __init__(self, rows, columns, loadpath, latent_dim = 128, discriminator_extra_steps=5, gp_weight=10.0):
        super(WGAN_GP, self).__init__()

        self.rows = rows
        self.columns = columns
        self.channels = 1
        self.img_shape = (self.rows, self.columns, self.channels)

        self.latent_dim = latent_dim
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight

        self.hidden_units = 800
        self.intermediate_units = 3072
        self.attention_heads = 10
        self.pos_embedding = self.add_weight(
            "position_embedding",
            shape=[self.rows, self.hidden_units],
            initializer=initializers.TruncatedNormal(stddev=0.02),
            dtype=tf.float32)

        self._start_epoch_index = 0
        json_file = os.path.join(loadpath, "wgan_gp.json")
        if os.path.isfile(json_file):
            print("########## Load json from %s ##########" % json_file)
            with open(json_file, 'r') as f:
                json_str = f.read()
                json_data = json.loads(json_str)
                self._start_epoch_index = json_data['finished_epochs']
                print("########## Json str: %s  ##########" % json_str)

        # discriminator
        self.discriminator = None
        if loadpath is not None:
            discriminator_path = os.path.join(loadpath, "discriminator")
            if os.path.isdir(discriminator_path):
                print("########## Load discriminator from %s ##########" % discriminator_path)
                self.discriminator = keras.models.load_model(discriminator_path)

        if self.discriminator is None:
            print("########## Create discriminator ##########")
            self.discriminator = self.get_discriminator_model()
            self.discriminator.summary()

        self.d_optimizer = keras.optimizers.Adam(
            learning_rate=0.0002, beta_1=0.5, beta_2=0.9
        )
        self.d_loss_fn = WGAN_GP.discriminator_loss

        # generator
        self.generator = None
        if loadpath is not None:
            generator_path = os.path.join(loadpath, "generator")
            if os.path.isdir(generator_path):
                print("########## Load generator from %s ##########" % generator_path)
                self.generator = keras.models.load_model(generator_path)

        if self.generator is None:
            print("########## Create generator ##########")
            self.generator = self.get_generator_model()
            self.generator.summary()

        self.g_optimizer = keras.optimizers.Adam(
            learning_rate=0.0002, beta_1=0.5, beta_2=0.9
        )
        self.g_loss_fn = WGAN_GP.generator_loss

    @property
    def start_epoch_index(self):
        return self._start_epoch_index

    @start_epoch_index.setter
    def start_epoch_index(self, value):
        raise ValueError

    def dense_block(self,
                  x,
                  units,
                  activation,
                  use_bias=True,
                  use_bn=False,
                  use_dropout=False,
                  drop_value=0.2
                  ):
        x = layers.Dense(units, activation= activation, use_bias=use_bias)(x)
        if use_bn:
            x = layers.BatchNormalization()(x)
        if use_dropout:
            x = layers.Dropout(drop_value)(x)
        return x

    # ---------------------
    #  Discriminator
    # ---------------------

    def conv_block(self,
            x,
            filters,
            activation,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            use_bias=True,
            use_bn=False,
            use_dropout=False,
            drop_value=0.5,
    ):
        x = layers.Conv2D(
            filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias
        )(x)
        if use_bn:
            x = layers.BatchNormalization()(x)
        if activation:
            x = activation(x)
        if use_dropout:
            x = layers.Dropout(drop_value)(x)
        return x

    def get_discriminator_model(self):
        img_input = layers.Input(shape=self.img_shape)
        # (None, 600, 75, 1)

        x = img_input
        x = layers.Reshape((self.rows, self.columns))(x)
        x = self.dense_block(x, self.columns * 16, layers.LeakyReLU(0.2),
                           use_bias=True, use_bn=False, use_dropout=True, drop_value=0.2)
        x = self.dense_block(x, self.columns * 8, layers.LeakyReLU(0.2),
                           use_bias=True, use_bn=False, use_dropout=True, drop_value=0.2)
        x = self.dense_block(x, self.columns * 4, layers.LeakyReLU(0.2),
                           use_bias=True, use_bn=False, use_dropout=True, drop_value=0.2)
        x = self.dense_block(x, self.columns * 2, layers.LeakyReLU(0.2),
                           use_bias=True, use_bn=False, use_dropout=True, drop_value=0.2)

        x = layers.Flatten()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(1)(x)

        d_model = keras.models.Model(img_input, x, name="discriminator")
        return d_model

    @staticmethod
    def discriminator_loss(real_img, fake_img):
        real_loss = tf.reduce_mean(real_img)
        fake_loss = tf.reduce_mean(fake_img)
        return fake_loss - real_loss

    # ---------------------
    #  Generator
    # ---------------------

    def upsample_block(self,
            x,
            filters,
            activation,
            kernel_size=(3, 3),
            strides=(1, 1),
            up_size=(2, 2),
            padding="same",
            use_bn=False,
            use_bias=True,
            use_dropout=False,
            drop_value=0.3,
    ):
        x = layers.UpSampling2D(up_size)(x)
        x = layers.Conv2D(
            filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias
        )(x)
        if use_bn:
            x = layers.BatchNormalization()(x)
        if activation:
            x = activation(x)
        if use_dropout:
            x = layers.Dropout(drop_value)(x)
        return x

    def get_generator_model(self):
        noise = layers.Input(shape=(self.latent_dim,))
        # (None, 128)

        x = layers.Dense(self.rows * self.columns * 8, use_bias=False)(noise)
        x = layers.Reshape((self.rows, self.columns * 8))(x)
        x = self.dense_block(x, self.columns * 8, layers.LeakyReLU(0.2),
                           use_bias=True, use_bn=False, use_dropout=True, drop_value=0.2)
        x = self.dense_block(x, self.columns * 4, layers.LeakyReLU(0.2),
                           use_bias=True, use_bn=False, use_dropout=True, drop_value=0.2)
        x = self.dense_block(x, self.columns * 2, layers.LeakyReLU(0.2),
                           use_bias=True, use_bn=False, use_dropout=True, drop_value=0.2)
        x = self.dense_block(x, self.columns, layers.LeakyReLU(0.2),
                           use_bias=True, use_bn=False, use_dropout=True, drop_value=0.2)

        x = layers.Reshape((self.rows, self.columns, self.channels))(x)

        g_model = keras.models.Model(noise, x, name="generator")
        return g_model

    @staticmethod
    def generator_loss(fake_img):
        return -tf.reduce_mean(fake_img)

    # ---------------------
    #  Transformer
    # ---------------------

    def linear_embedding(self, x):
        x = layers.Dense(self.hidden_units)(x)
        return x

    def position_embedding(self, x):
        return x + self.pos_embedding

    def transformer(self,
                    x,
                    num_hidden_layers=12,
                    hidden_units=800,
                    intermediate_units=3072,
                    attention_heads=10):
        for _ in range(num_hidden_layers):
            tmp = self.norm(x)
            tmp = self.attention(tmp, hidden_units=hidden_units, attention_heads=attention_heads)
            x = x + tmp

            tmp = self.norm(x)
            tmp = self.mlp(tmp, hidden_units=hidden_units, intermediate_units=intermediate_units)
            x = x + tmp
        return x

    @staticmethod
    def gelu(x):
        """Gaussian Error Linear Unit.
        This is a smoother version of the RELU.
        Original paper: https://arxiv.org/abs/1606.08415
        Args:
          x: float Tensor to perform activation.
        Returns:
          `x` with the GELU activation applied.
        """
        cdf = 0.5 * (1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
        return x * cdf

    def mlp(self, x, hidden_units=800, intermediate_units=3072):
        out = layers.Dense(intermediate_units, activation=WGAN_GP.gelu)(x)
        out = layers.Dense(hidden_units)(out)
        return out

    def attention(self, x, hidden_units=800, attention_heads=10):
        qkv = layers.Dense(hidden_units * 3, use_bias=False)(x)
        qkv = Rearrange("b n (qkv h d) -> qkv b h n d", qkv=3, h=attention_heads)(qkv)
        q = qkv[0]
        k = qkv[1]
        v = qkv[2]
        dots = tf.einsum("bhid,bhjd->bhij", q, k) * (hidden_units ** -0.5)
        attn = tf.nn.softmax(dots, axis=-1)
        out = tf.einsum("bhij,bhjd->bhid", attn, v)
        out = Rearrange("b h n d -> b n (h d)")(out)
        out = layers.Dense(hidden_units)(out)
        return out

    def norm(self, x):
        out = layers.LayerNormalization(epsilon=1e-5)(x)
        return out

    # ---------------------
    #  keras.Model
    # ---------------------

    def compile(self):
        super(WGAN_GP, self).compile()
        self.discriminator.compile()
        self.generator.compile()

    def gradient_penalty(self, batch_size, real_images, fake_images):
        """ Calculates the gradient penalty.
        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, real_images):
        if isinstance(real_images, tuple):
            real_images = real_images[0]

        # Get the batch size
        batch_size = tf.shape(real_images)[0]

        # For each batch, we are going to perform the
        # following steps as laid out in the original paper:
        # 1. Train the generator and get the generator loss
        # 2. Train the discriminator and get the discriminator loss
        # 3. Calculate the gradient penalty
        # 4. Multiply this gradient penalty with a constant weight factor
        # 5. Add the gradient penalty to the discriminator loss
        # 6. Return the generator and discriminator losses as a loss dictionary

        # ---------------------
        #  Train discriminator
        # ---------------------
        # The original paper recommends training
        # the discriminator for `x` more steps (typically 5) as compared to
        # one step of the generator. Here we will train it for 3 extra steps
        # as compared to 5 to reduce the training time.
        for i in range(self.d_steps):
            # Get the latent vector
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim)
            )
            with tf.GradientTape() as tape:
                # Generate fake images from the latent vector
                fake_images = self.generator(random_latent_vectors, training=True)
                # Get the logits for the fake images
                fake_logits = self.discriminator(fake_images, training=True)
                # Get the logits for the real images
                real_logits = self.discriminator(real_images, training=True)

                # Calculate the discriminator loss using the fake and real image logits
                d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, real_images, fake_images)
                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + gp * self.gp_weight

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )

        # ---------------------
        #  Train generator
        # ---------------------
        # Get the latent vector
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = self.generator(random_latent_vectors, training=True)
            # Get the discriminator logits for fake images
            gen_img_logits = self.discriminator(generated_images, training=True)
            # Calculate the generator loss
            g_loss = self.g_loss_fn(gen_img_logits)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        return {"d_loss": d_loss, "g_loss": g_loss}

    def save(self,
        finished_epochs,
        filepath,
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None,
        save_traces=True
    ):
        print("########## Save mode, finished_epochs = %d ##########" % finished_epochs)
        if not os.path.isdir(filepath):
            os.makedirs(filepath)

        json_data = {'finished_epochs' : finished_epochs}
        json_str = json.dumps(json_data)
        json_file = os.path.join(filepath, "wgan_gp.json")
        with open(json_file, 'w') as f:
            f.write(json_str)

        discriminator_path = os.path.join(filepath, "discriminator")
        if not os.path.isdir(discriminator_path):
            os.makedirs(discriminator_path)
        self.discriminator.save(discriminator_path,
                                overwrite=overwrite,
                                include_optimizer=include_optimizer,
                                save_format=save_format,
                                signatures=signatures,
                                options=options,
                                save_traces=save_traces)

        generator_path = os.path.join(filepath, "generator")
        if not os.path.isdir(generator_path):
            os.makedirs(generator_path)
        self.generator.save(generator_path,
                                overwrite=overwrite,
                                include_optimizer=include_optimizer,
                                save_format=save_format,
                                signatures=signatures,
                                options=options,
                                save_traces=save_traces)

