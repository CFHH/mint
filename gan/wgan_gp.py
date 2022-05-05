from __future__ import print_function, division

import json

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os


class WGAN_GP(keras.Model):
    def __init__(self, loadpath, latent_dim = 128, discriminator_extra_steps=5, gp_weight=10.0):
        super(WGAN_GP, self).__init__()
        self.latent_dim = latent_dim
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight

        self.rows = 28
        self.columns = 28
        self.channels = 1
        self.img_shape = (self.rows, self.columns, self.channels)

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
            #self.discriminator.summary()

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
            # self.generator.summary()

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
        x = activation(x)
        if use_dropout:
            x = layers.Dropout(drop_value)(x)
        return x

    def get_discriminator_model(self):
        img_input = layers.Input(shape=self.img_shape)
        # (None, 28, 28, 1)

        # ZeroPadding2D(padding), padding = ((top_pad, bottom_pad), (left_pad, right_pad))
        # 这里是上下左右各填1个零变成(32, 32, 1)，(2, 2)=((1, 1), (1, 1))
        x = layers.ZeroPadding2D((2, 2))(img_input)
        # ->(None, 32, 32, 1)

        x = self.conv_block(
            x,
            64,
            kernel_size=(5, 5),
            strides=(2, 2),
            use_bn=False,
            use_bias=True,
            activation=layers.LeakyReLU(0.2),
            use_dropout=False,
            drop_value=0.3,
        )
        # ->(None, 16, 16, 64)

        x = self.conv_block(
            x,
            128,
            kernel_size=(5, 5),
            strides=(2, 2),
            use_bn=False,
            activation=layers.LeakyReLU(0.2),
            use_bias=True,
            use_dropout=True,
            drop_value=0.3,
        )
        # ->(None, 8, 8, 128)

        x = self.conv_block(
            x,
            256,
            kernel_size=(5, 5),
            strides=(2, 2),
            use_bn=False,
            activation=layers.LeakyReLU(0.2),
            use_bias=True,
            use_dropout=True,
            drop_value=0.3,
        )
        # ->(None, 4, 4, 256)

        x = self.conv_block(
            x,
            512,
            kernel_size=(5, 5),
            strides=(2, 2),
            use_bn=False,
            activation=layers.LeakyReLU(0.2),
            use_bias=True,
            use_dropout=False,
            drop_value=0.3,
        )
        # ->(None, 2, 2, 512)

        x = layers.Flatten()(x)
        # ->(None, 2048)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(1)(x)
        # ->(None, 1)

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

        x = layers.Dense(4 * 4 * 256, use_bias=False)(noise)
        # ->(None, 4096)

        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Reshape((4, 4, 256))(x)
        # ->(None, 4, 4, 256)

        x = self.upsample_block(
            x,
            128,
            layers.LeakyReLU(0.2),
            strides=(1, 1),
            use_bias=False,
            use_bn=True,
            padding="same",
            use_dropout=False,
        )
        # ->(None, 8, 8, 256)->(None, 8, 8, 128)

        x = self.upsample_block(
            x,
            64,
            layers.LeakyReLU(0.2),
            strides=(1, 1),
            use_bias=False,
            use_bn=True,
            padding="same",
            use_dropout=False,
        )
        # ->(None, 16, 16, 128)->(None, 16, 16, 64)

        x = self.upsample_block(
            x,
            1,
            layers.Activation("tanh"),
            strides=(1, 1),
            use_bias=False,
            use_bn=True
        )
        # ->(None, 32, 32, 64)->(None, 32, 32, 1)

        # 同ZeroPadding2D
        x = layers.Cropping2D((2, 2))(x)
        # ->(None, 28, 28, 1)

        g_model = keras.models.Model(noise, x, name="generator")
        return g_model

    @staticmethod
    def generator_loss(fake_img):
        return -tf.reduce_mean(fake_img)

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
        print(real_images.shape)
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




'''
if __name__ == '__main__':
    noise_dim = 128
    batch_size = 32
    epochs = 1000000

    wgan_gp = WGAN_GP(latent_dim = noise_dim, discriminator_extra_steps=3)
    wgan_gp.compile()

    cbk = GanMonitor(num_img=3, latent_dim=noise_dim)

    # TODO tf.DataSet
    train_dataset = None

    # TODO 改成不停的调用 wgan_gp.train_step() 或者能用orbit吗？
    wgan_gp.fit(train_dataset, batch_size=batch_size, epochs=epochs, callbacks=[cbk])

    # TODO 保存模型数据
'''
