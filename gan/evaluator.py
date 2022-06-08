from __future__ import print_function, division
from absl import app
from absl import flags
import numpy as np
import tensorflow as tf
import math
import os
import wgan_gp

FLAGS = flags.FLAGS
flags.DEFINE_integer('motion_frame_length', 480, 'motion frame length') # 需要和trainer的一致
flags.DEFINE_integer('motion_dimension', 75, 'motion frame length') # 需要和trainer的一致
flags.DEFINE_integer('noise_dim', 128, 'noise dim')
flags.DEFINE_string('mode_dir', "./model", '保存模型的路径')
flags.DEFINE_string('sample_dir', "./samples_eval", '生成数据的路径')


def get_model():
    save_path = "%s/%s" % (FLAGS.mode_dir, "tmp")
    mode = wgan_gp.WGAN_GP(rows=FLAGS.motion_frame_length, columns=FLAGS.motion_dimension, loadpath=save_path, latent_dim=FLAGS.noise_dim, discriminator_extra_steps=3)
    return mode


def evaluate():
    # 加载模型
    model = get_model()
    if model is None:
        print("model is None!!!!!!!!!!")
        return
    model.compile()

    # 生成数据
    generate_num = 100
    if not os.path.isdir(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)
    random_latent_vectors = tf.random.normal(shape=(generate_num, FLAGS.noise_dim))
    generated_datas = model.generator(random_latent_vectors)
    two_pi = math.pi * 2
    generated_datas = generated_datas * two_pi
    for i in range(generate_num):
        file_data = generated_datas[i]
        file_name = "eval_%03d" % i
        save_path = os.path.join(FLAGS.sample_dir, file_name)
        np.save(save_path, file_data)


def main(_):
    evaluate()

if __name__ == '__main__':
    # nohup python -u trainer.py > log_train.log 2>&1 &
    app.run(main)