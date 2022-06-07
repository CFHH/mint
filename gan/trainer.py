from __future__ import print_function, division
from absl import app
from absl import flags
import numpy as np
import tensorflow as tf
import math
import os
import wgan_gp
import dataset_loader

FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 8, 'batch size')
flags.DEFINE_integer('noise_dim', 128, 'noise dim')
flags.DEFINE_integer('epochs', 99999999, '迭代总次数')
flags.DEFINE_string('mode_dir', "./model", '保存模型的路径')
flags.DEFINE_string('sample_dir', "./samples", '生成数据的路径')


def get_model():
    save_path = "%s/%s" % (FLAGS.mode_dir, "tmp")
    mode = wgan_gp.WGAN_GP(loadpath=save_path, latent_dim=FLAGS.noise_dim, discriminator_extra_steps=3)
    return mode


def sample_datas(model, finished_epochs, num_img, latent_dim):
    if not os.path.isdir(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)
    random_latent_vectors = tf.random.normal(shape=(num_img, latent_dim))
    generated_datas = model.generator(random_latent_vectors)
    two_pi = math.pi * 2
    generated_datas = generated_datas * two_pi
    for i in range(num_img):
        file_data = generated_datas[i]
        file_name = "gen_{epoch}_{i}".format(epoch=finished_epochs, i=i)
        save_path = os.path.join(FLAGS.sample_dir, file_name)
        np.save(save_path, file_data)


def train():
    # 加载模型
    model = get_model()
    if model is None:
        print("model is None!!!!!!!!!!")
        return
    model.compile()

    # 加载数据集
    train_dataset = dataset_loader.load_dataset(FLAGS.batch_size, "../data/gan_with_trans/gan_train_tfrecord-*")
    train_iter = tf.nest.map_structure(iter, train_dataset)

    # 训练
    two_pi = math.pi * 2
    for epoch in range(model.start_epoch_index, FLAGS.epochs):
        #print("---------- train epoch(%d) ----------" % epoch)
        data = next(train_iter)
        real_data = data["motion_input"]              # (batch_size, 120, 72)
        real_data = np.expand_dims(real_data, axis=3) # (batch_size, 120, 72, 1)
        real_data = real_data / two_pi
        loss = model.train_step(real_data)
        """
        10万步，永久保存模型
        1万步，临时保存模型
        1万步，生成数据
        """
        finished_epochs = epoch + 1
        if finished_epochs % 100 == 0:
            print("---------- train epoch(%d), discriminator loss is %f, generator loss is %f----------" % (finished_epochs, loss["d_loss"], loss["g_loss"]))

        if finished_epochs % 5000 == 0:
            save_path = "%s/%08d" % (FLAGS.mode_dir, finished_epochs)
            model.save(finished_epochs, save_path)
        elif finished_epochs % 1000 == 0:
            save_path = "%s/%s" % (FLAGS.mode_dir, "tmp")
            model.save(finished_epochs, save_path)

        if finished_epochs % 5000 == 0:
            sample_datas(model, finished_epochs, 5, FLAGS.noise_dim)


def main(_):
    train()

if __name__ == '__main__':
    # nohup python -u trainer.py > log_train.log 2>&1 &
    app.run(main)