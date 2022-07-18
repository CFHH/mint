from absl import app
from absl import flags
from absl import logging
import os
import numpy as np
import math

from aist_plusplus.loader import AISTDataset
import smpl_bvh_writer

# python bvh_dataset_generater.py --bvh_data_path=../data/bvh_scale/ --scale100=True
# python bvh_dataset_generater.py --bvh_data_path=../data/bvh_noscale/ --scale100=False

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'anno_dir', '../data/aist_plusplus_final',
    'Path to the AIST++ annotation files.')
flags.DEFINE_string(
    'bvh_data_path', '../data/bvh_scale/', #路径需要带/，不然makedirs创建不了
    'Output path for the bvh files.')
flags.DEFINE_boolean(
    'scale100', True, #默认True是因为放大了才能在bvhacker里查看
    'scale position and offset')

def main(_):
    os.makedirs(os.path.dirname(FLAGS.bvh_data_path), exist_ok=True)

    seq_names = []

    seq_names += np.loadtxt(
        os.path.join(FLAGS.anno_dir, "splits/crossmodal_train.txt"), dtype=str
    ).tolist()

    seq_names += np.loadtxt(
        os.path.join(FLAGS.anno_dir, "splits/crossmodal_val.txt"), dtype=str
    ).tolist()

    seq_names += np.loadtxt(
        os.path.join(FLAGS.anno_dir, "splits/crossmodal_test.txt"), dtype=str
    ).tolist()

    ignore_list = np.loadtxt(
        os.path.join(FLAGS.anno_dir, "ignore_list.txt"), dtype=str
    ).tolist()

    seq_names = [name for name in seq_names if name not in ignore_list]

    dataset = AISTDataset(FLAGS.anno_dir)

    n_samples = len(seq_names)
    for i, seq_name in enumerate(seq_names):
        logging.info("processing %d / %d, %s" % (i + 1, n_samples, seq_name))

        smpl_poses, smpl_scaling, smpl_trans = AISTDataset.load_motion(dataset.motion_dir, seq_name)
        # smpl_trans.shape = (frame,3), positions的x和z在0附近，y在2.0左右
        positions = smpl_trans / smpl_scaling

        # smpl_poses.shape = (frame,72)
        frames, dim = smpl_poses.shape
        assert dim == 72
        rotations = smpl_poses * (180/math.pi) #换成角度
        rotations = rotations.reshape(frames, 24, 3)
        #rotations = rotations[:, smpl_bvh_writer.ROTATION_SEQ, :] #写入时再转换吧

        save_path = os.path.join(FLAGS.bvh_data_path, '%s.bvh' % seq_name)
        frametime = 1.0/60 #帧时间，这个库是每秒60帧
        smpl_bvh_writer.save_motion_as_bvh(save_path, positions, rotations, frametime, scale100=FLAGS.scale100)

if __name__ == '__main__':
  app.run(main)