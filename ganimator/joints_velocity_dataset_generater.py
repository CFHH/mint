from absl import app
from absl import flags
from absl import logging
import os
import numpy as np
import math
import glob
import tqdm
import re
import torch
import json
import bvh_loader
import smpl_bvh_writer
from tools.kinematics import ForwardKinematicsJoint
from tools.Quaternions import Quaternions

FLAGS = flags.FLAGS
flags.DEFINE_string('motion_data_path', '../data/bvh_scale/', 'motion data path')
flags.DEFINE_string('music_data_file', '../data/bvh_scale_split/music_feature.dat', 'music feature file')
flags.DEFINE_boolean('scaled', True, '动作是否已经放大了100倍')
flags.DEFINE_string('save_path', '../data/velocity/', 'velocity and angular_velocity')

def joint_position(fk, root_positions, rotations):
    root_positions = torch.tensor(root_positions, dtype=torch.float)
    rotations = Quaternions.from_euler(np.radians(rotations)).qs
    rotations = torch.tensor(rotations, dtype=torch.float)
    all_positions = fk.forward(rotations, root_positions) # 计算各骨骼节点的世界坐标位置(frame, 24, 3)
    return all_positions.numpy()

def process(filename, music_data):
    """
    names         : (24,)，各骨骼节点的名字，24按bvh中出场次序（下同）
    parents       : (24,)，各骨骼节点的父节点索引
    offsets       : (24,3)，各骨骼节点在初始状态下的对父节点的位移（单位是米），也就是bvh中的OFFSET
    root_positions: (帧数,3)，各帧根骨骼的位移（单位是米）
    rotations     : (帧数,24,3)，各帧各骨骼节点的欧拉角（单位是角度）
    以上数字数据的类型都是numpy数组
    """
    names, parents, offsets, root_positions, rotations, frametime = bvh_loader.load_bvh_motion(filename, FLAGS.scaled)
    parents_temp = tuple(parents)
    offsets_temp = torch.tensor(offsets, dtype=torch.float)
    fk = ForwardKinematicsJoint(parents_temp, offsets_temp)

    """
    positions: (帧数,24,3)，各帧各节点的世界坐标
    """
    positions = joint_position(fk, root_positions, rotations)

    """
    velocity: (帧数,24)，各帧各节点的速度（单位是米/秒）
    """
    velocity = positions[1:, :, :] - positions[:-1, :, :] #(帧数-1,24,3)
    first_frame = np.zeros_like(positions[0:1:,]) #(1,24,3)
    velocity = np.concatenate([first_frame, velocity], axis=0) #(帧数,24,3)
    velocity *= 60.0 # 60fps
    velocity = np.linalg.norm(velocity, axis=-1) #(帧数,24)

    """
    frame_offsets: (帧数,23,3)，各帧各节点(除根节点)相对于父节点的偏移
    bone_length: (23,)，各节点(除根节点)距离父节点的长度
    """
    frame_offsets = positions[:, 1:, :] - positions[:, parents[1:], :]

    bone_length = smpl_bvh_writer.SMPL_JOINTS_OFFSETS[smpl_bvh_writer.ROTATION_SEQ]
    bone_length = bone_length[1:, :] - bone_length[parents[1:], :]
    bone_length = np.linalg.norm(bone_length, axis=-1)

    """
    angular_velocity: (帧数,24)，各帧各节点的角速度（单位是弧度/秒）
    """
    angular_velocity = frame_offsets[1:, :, :] - frame_offsets[:-1, :, :] #(帧数-1,23,3)
    first_frame = np.zeros_like(frame_offsets[0:1:, ])  # (1,23,3)
    angular_velocity = np.concatenate([first_frame, angular_velocity], axis=0)  # (帧数,23,3)
    angular_velocity = np.linalg.norm(angular_velocity, axis=-1) # (帧数,23)
    angular_velocity *= 60.0  # 60fps，先模再60，更接近弧长
    angular_velocity = angular_velocity / bone_length # (帧数,23)
    first_bone = np.zeros_like(angular_velocity[:, 0:1])  # (帧数, 1)
    angular_velocity = np.concatenate([first_bone, angular_velocity], axis=1)  # (帧数,24)

    """
    音乐数据
    """
    motion_name = os.path.basename(filename)
    motion_name = motion_name.split('.')[0]
    music_feature = music_data[motion_name]
    strength = np.array(music_feature['strength'], dtype=np.float32)
    peak_idx = music_feature['peak_idx']
    beat_idx = music_feature['beat_idx']
    peak_onehot = np.zeros_like(strength, dtype=np.float32)
    peak_onehot[peak_idx] = 1.0
    beat_onehot = np.zeros_like(strength, dtype=np.float32)
    beat_onehot[beat_idx] = 1.0

    strength = strength[:, np.newaxis]
    peak_onehot = peak_onehot[:, np.newaxis]
    beat_onehot = beat_onehot[:, np.newaxis]
    music_feature = np.concatenate([strength, peak_onehot, beat_onehot], axis=-1) # (帧数,3)

    """
    合并一下，输出到文件
    """
    motion_frames = velocity.shape[0]
    music_frames = music_feature.shape[0]
    final_frames = min(motion_frames, music_frames)
    if motion_frames > final_frames:
        velocity = velocity[0:final_frames,]
        angular_velocity = angular_velocity[0:final_frames,]
    else:
        music_feature = music_feature[0:final_frames,]

    final_data = np.concatenate([velocity, angular_velocity, music_feature], axis=-1) # (帧数, 24+24+3)

    save_file = os.path.join(FLAGS.save_path, motion_name)
    np.save(save_file, final_data)
    # 测试加载 data = np.load('../data/velocity/gWA_sFM_cAll_d25_mWA4_ch05.npy')
    return

def main(_):
    music_data = None
    if os.path.isfile(FLAGS.music_data_file):
        with open(FLAGS.music_data_file, 'r') as f:
            json_str = f.read()
            music_data = json.loads(json_str)

    # 调试用
    #process("../data/bvh_scale/gWA_sFM_cAll_d25_mWA4_ch05.bvh", music_data)

    motion_files = glob.glob(os.path.join(FLAGS.motion_data_path, "*.bvh"))
    for file in tqdm.tqdm(motion_files):
        print("Process %s" % file)
        process(file, music_data)
    return

if __name__ == '__main__':
    app.run(main)