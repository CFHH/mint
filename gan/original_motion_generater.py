from absl import app
from absl import flags
from absl import logging

import os
import random
import numpy as np
from scipy.spatial.transform import Rotation as R
import librosa
from aist_plusplus.loader import AISTDataset

import tensorflow as tf


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'anno_dir', '../data/aist_plusplus_final',
    'Path to the AIST++ annotation files.')
flags.DEFINE_string(
    'audio_dir', 'D:/mint-music/',
    'Path to the AIST wav files.')
flags.DEFINE_string(
    'audio_cache_dir', '../data/tmp/aist_audio_feats',
    'Path to cache dictionary for audio features.')
flags.DEFINE_enum(
    'split', 'train', ['train', 'testval'],
    'Whether do training set or testval set.')
flags.DEFINE_string(
    'tfrecord_path', '../data/gan/gan_train_tfrecord',
    'Output path for the tfrecord files.')
flags.DEFINE_string(
    'original_motion_data_path', '../data/original_motion_data/',
    'Output path for the tfrecord files.')


def main(_):
    os.makedirs(os.path.dirname(FLAGS.tfrecord_path), exist_ok=True)

    # create list
    seq_names = []
    if "train" in FLAGS.split:
        seq_names += np.loadtxt(
            os.path.join(FLAGS.anno_dir, "splits/crossmodal_train.txt"), dtype=str
        ).tolist()
    if "val" in FLAGS.split:
        seq_names += np.loadtxt(
            os.path.join(FLAGS.anno_dir, "splits/crossmodal_val.txt"), dtype=str
        ).tolist()
    if "test" in FLAGS.split:
        seq_names += np.loadtxt(
            os.path.join(FLAGS.anno_dir, "splits/crossmodal_test.txt"), dtype=str
        ).tolist()
    ignore_list = np.loadtxt(
        os.path.join(FLAGS.anno_dir, "ignore_list.txt"), dtype=str
    ).tolist()
    seq_names = [name for name in seq_names if name not in ignore_list]
    
    # load data
    dataset = AISTDataset(FLAGS.anno_dir)
    n_samples = len(seq_names)
    for i, seq_name in enumerate(seq_names):
        logging.info("processing %d / %d, %s" % (i + 1, n_samples, seq_name))

        smpl_poses, smpl_scaling, smpl_trans = AISTDataset.load_motion(dataset.motion_dir, seq_name)
        smpl_trans /= smpl_scaling
        smpl_motion = np.concatenate([smpl_trans, smpl_poses], axis=-1)

        file_data = smpl_motion
        file_name = seq_name
        save_path = os.path.join(FLAGS.original_motion_data_path, file_name)
        np.save(save_path, file_data)

    # If testval, also test on un-paired data
    if FLAGS.split == "testval":
        logging.info("Also add un-paired motion-music data for testing.")
        for i, seq_name in enumerate(seq_names * 10):
            logging.info("processing %d / %d, %s" % (i + 1, n_samples * 10, seq_name))

            smpl_poses, smpl_scaling, smpl_trans = AISTDataset.load_motion(dataset.motion_dir, seq_name)
            smpl_trans /= smpl_scaling
            smpl_motion = np.concatenate([smpl_trans, smpl_poses], axis=-1)

            file_data = smpl_motion
            file_name = seq_name
            save_path = os.path.join(FLAGS.original_motion_data_path, file_name)
            np.save(save_path, file_data)

if __name__ == '__main__':
  app.run(main)