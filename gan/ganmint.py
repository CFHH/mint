from absl import app
from absl import flags
import tensorflow as tf
import inputs

'''TODO
keras用tf的数据集
keras用GPU
训练结果的保存和载入
能不能使用strategy、trainer、controller
'''

'''
运行参数
'''
FLAGS = flags.FLAGS

'''
配置
'''
def get_configs():
    return None


'''
'''
def get_distribution_strategy():
    strategy = tf.distribute.MirroredStrategy()
    return strategy


'''
数据集
def get_dataset_fn():
    def get_dataset_fn_inner(input_context=None):
        del input_context
        dataset = None
        return dataset
    return get_dataset_fn_inner
'''


'''
训练模型
'''
def get_model():
    discriminator = None
    generator = None
    return generator, discriminator




def train():
    train_dataset = inputs.get_dataset_fn(3, "../data/tf_sstables/aist_generation_train_v2_tfrecord-*")
    train_iter = tf.nest.map_structure(iter, train_dataset)
    for _ in range(100):
        data = next(train_iter)
        #print(data["motion_input"].shape)
        print(data["motion_input"][0][100][100:110])



def main(_):
  train()

if __name__ == '__main__':
  app.run(main)