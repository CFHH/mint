from __future__ import print_function, division
from absl import app
from absl import flags
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import wgan_gp
import dataset_loader

FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 32, 'batch size')
flags.DEFINE_integer('noise_dim', 128, 'noise dim')
flags.DEFINE_integer('epochs', 1000000, '迭代总次数')
flags.DEFINE_integer('sample_interval', 500, '迭代多少次保存一回模型')
flags.DEFINE_string('mode_dir', "./model", '保存模型的路径')
flags.DEFINE_string('sample_dir', "./samples", '生成数据的路径')
flags.DEFINE_integer('load_epoch', 0, '1表示加载模型')


def get_model(load_epoch):
    mode = wgan_gp.WGAN_GP(loadpath=FLAGS.mode_dir, latent_dim=FLAGS.noise_dim, discriminator_extra_steps=3)
    return mode


class GanMonitor(keras.callbacks.Callback):
    def __init__(self, num_img=6, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        epoch = self.model.start_epoch_index + epoch
        print("GanMonitor.on_epoch_end %d" % epoch)
        self.save_model(epoch)
        self.sample_images(epoch)

    def save_model(self, epoch):
        finished_epochs = epoch + 1
        self.model.save(finished_epochs, FLAGS.mode_dir)

    def sample_images(self, epoch):
        if not os.path.isdir(FLAGS.sample_dir):
            os.makedirs(FLAGS.sample_dir)

        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
        generated_images = self.model.generator(random_latent_vectors)
        generated_images = (generated_images * 127.5) + 127.5
        for i in range(self.num_img):
            img = generated_images[i].numpy()
            img = keras.preprocessing.image.array_to_img(img)
            image_name = "generated_img_{i}_{epoch}.png".format(i=i, epoch=epoch)
            img.save(os.path.join(FLAGS.sample_dir, image_name))


def train():
    #  train_images：60000 28*28
    IMG_SHAPE = (28, 28, 1)
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    # Reshape each sample to (28, 28, 1) and normalize the pixel values in the [-1, 1] range
    train_images = train_images.reshape(train_images.shape[0], *IMG_SHAPE).astype("float32")
    train_images = (train_images - 127.5) / 127.5

    train_images = train_images[0:3200]

    load_epoch = FLAGS.load_epoch

    mode = get_model(load_epoch)
    if mode is None:
        print("mode is None!")
        return
    mode.compile()

    cbk = GanMonitor(num_img=3, latent_dim=FLAGS.noise_dim)

    mode.fit(train_images, batch_size=FLAGS.batch_size, epochs=FLAGS.epochs, callbacks=[cbk])

def main(_):
    '''测试dataset
    batch_size = 7
    train_dataset = dataset_loader.load_dataset(batch_size, "../data/gan/gan_train_tfrecord-*")
    train_iter = tf.nest.map_structure(iter, train_dataset)
    data = next(train_iter)
    print(data)
    '''
    train()

if __name__ == '__main__':
    app.run(main)