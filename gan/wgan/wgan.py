from __future__ import print_function, division

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import RMSprop
import tensorflow.keras.backend as K

import inputs
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class WGAN():
    def __init__(self):
        # 输入参数
        self.rows = 120    # 动作的帧数 120，120/4 = 30
        self.columns = 220 # 动作的特征数 225->220（24*9+3+1），220/4 = 55
        self.channels = 1
        self.img_shape = (self.rows, self.columns, self.channels)
        self.latent_dim = 100

        # 这些参数和优化器是根据论文来的
        self.n_critic = 5               # the number of iterations of the critic per generator iteration
        self.clip_value = 0.01          # clipping parameter
        optimizer = RMSprop(lr=0.00005) # learning rate

        # 建模critic
        self.critic = self.build_critic()
        self.critic.compile(loss=self.wasserstein_loss,
            optimizer=optimizer,
            metrics=['accuracy'])

        # 建模generator
        self.generator = self.build_generator()

        # 建模combined
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)
        self.critic.trainable = False # combined模型只训练generator【不知道为什么这样起作用】
        valid = self.critic(img)
        self.combined = Model(z, valid)
        self.combined.compile(loss=self.wasserstein_loss,
            optimizer=optimizer,
            metrics=['accuracy'])

    def wasserstein_loss(self, y_true, y_pred):
        #论文中的损失函数
        return K.mean(y_true * y_pred)

    def build_generator(self):
        model = Sequential()
        # (None, 100) 输入
        model.add(Dense(128 * 30 * 55, activation="relu", input_dim=self.latent_dim))
        # (None, 211200) 参数21331200
        model.add(Reshape((30, 55, 128)))
        # (None, 30, 55, 128)
        model.add(UpSampling2D())
        # (None, 60, 110, 128)
        model.add(Conv2D(128, kernel_size=4, padding="same"))
        # (None, 60, 110, 128) 参数262272
        model.add(BatchNormalization(momentum=0.8))
        # (None, 60, 110, 128) 参数512
        model.add(Activation("relu"))
        # (None, 60, 110, 128)
        model.add(UpSampling2D())
        # (None, 120, 220, 128)
        model.add(Conv2D(64, kernel_size=4, padding="same"))
        # (None, 120, 220, 64) 参数131136
        model.add(BatchNormalization(momentum=0.8))
        # (None, 120, 220, 64) 参数256
        model.add(Activation("relu"))
        # (None, 120, 220, 64)
        model.add(Conv2D(self.channels, kernel_size=4, padding="same"))
        # (None, 120, 220, 1) 参数1025
        model.add(Activation("tanh"))
        # (None, 120, 220, 1) 输出

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)
        return Model(noise, img)

    def build_critic(self):
        model = Sequential()
        # (None, 120, 220, 1) 输入
        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        # (None, 60, 110, 16) 参数160
        model.add(LeakyReLU(alpha=0.2))
        # (None, 60, 110, 16)
        model.add(Dropout(0.25))
        # (None, 60, 110, 16)
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        # (None, 30, 55, 32) 参数4640
        model.add(ZeroPadding2D(padding=((0,0),(0,1))))
        # (None, 30, 56, 32)
        model.add(BatchNormalization(momentum=0.8))
        # (None, 30, 56, 32) 参数128
        model.add(LeakyReLU(alpha=0.2))
        # (None, 30, 56, 32)
        model.add(Dropout(0.25))
        # (None, 30, 56, 32)
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        # (None, 15, 28, 64) 参数18496
        model.add(BatchNormalization(momentum=0.8))
        # (None, 15, 28, 64) 参数256
        model.add(LeakyReLU(alpha=0.2))
        # (None, 15, 28, 64)
        model.add(Dropout(0.25))
        # (None, 15, 28, 64)
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        # (None, 15, 28, 128) 参数73856
        model.add(BatchNormalization(momentum=0.8))
        # (None, 15, 28, 128) 参数512
        model.add(LeakyReLU(alpha=0.2))
        # (None, 15, 28, 128)
        model.add(Dropout(0.25))
        # (None, 15, 28, 128)
        model.add(Flatten())
        # (None, 53760)
        model.add(Dense(1))
        # (None, 1) 参数53761
        # 输出：-1表示valid也就是数据集的真图，1是fake也就是生成的假图

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)
        return Model(img, validity)

    def train(self, epochs, batch_size=7, sample_interval=50):
        train_dataset = inputs.get_dataset_fn(batch_size, "../data/tf_sstables/aist_generation_train_v2_tfrecord-*")
        train_iter = tf.nest.map_structure(iter, train_dataset)

        valid = -np.ones((batch_size, 1)) #用-1表示valid，不是1
        fake = np.ones((batch_size, 1))   #用1表示fake，不是0

        for epoch in range(epochs):
            for _ in range(self.n_critic): # self.n_critic = 5
                # ---------------------
                #  Train Discriminator
                # ---------------------
                data = next(train_iter)
                valid_motions = data["motion_input"]                  # (batch_size, 120, 220)
                valid_motions = np.expand_dims(valid_motions, axis=3) # (batch_size, 120, 220, 1)

                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                fake_motions = self.generator.predict(noise)

                d_loss_valid = self.critic.train_on_batch(valid_motions, valid)
                d_loss_fake = self.critic.train_on_batch(fake_motions, fake)
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_valid)

                # Clip critic weights
                for l in self.critic.layers:
                    weights = l.get_weights()
                    # np.clip(a, a_min, a_max, out=None)
                    # self.clip_value = 0.01
                    weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    l.set_weights(weights)

            # ---------------------
            #  Train Generator
            # ---------------------
            g_loss = self.combined.train_on_batch(noise, valid)

            print ("%d [D loss: %f] [G loss: %f]" % (epoch, 1 - d_loss[0], 1 - g_loss[0]))

            # ---------------------
            #  Sample
            # ---------------------
            if epoch % sample_interval == 0:
                self.save_model(epoch)
                self.sample_motions(epoch)

    def sample_motions(self, epoch):
        output_dir = './samples'
        gen_num = 3
        noise = np.random.normal(0, 1, (gen_num, self.latent_dim))
        gen_motions = self.generator.predict(noise)

        for i in range(gen_num):
          gen_motion = gen_motions[i]
          save_path = os.path.join(output_dir, "%04d_%02d.npy" % (epoch, i))
          np.save(save_path, gen_motion)

    def save_model(self, epoch):
        def save(model, epoch, model_name):
            model_path = "./model/%d_%s.json" % (epoch, model_name)
            weights_path = "./model/%d_%s_weights.hdf5" % (epoch, model_name)
            json_string = model.to_json()
            open(model_path, 'w').write(json_string)
            model.save_weights(weights_path)
        save(self.generator, epoch, "generator")
        save(self.critic, epoch, "critic")


if __name__ == '__main__':
    wgan = WGAN()
    wgan.train(epochs=1000000, batch_size=32, sample_interval=500)
