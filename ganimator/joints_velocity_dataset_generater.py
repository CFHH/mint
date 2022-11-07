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
    1、velocity: (帧数,24)，各帧各节点的绝对速度（单位是米/秒）
    """
    velocity = positions[1:, :, :] - positions[:-1, :, :] # (帧数-1,24,3)
    first_frame = np.zeros_like(positions[0:1,]) #(1,24,3)
    velocity = np.concatenate([first_frame, velocity], axis=0) #(帧数,24,3)
    velocity *= 60.0 # 60fps
    relative_velocity = velocity[:, 1:, :] - velocity[:, parents[1:], :] # (帧数,23,3)
    root_velocity = velocity[:, 0:1, :].copy()# (帧数,1,3)
    acceleration = velocity[1:, :, :] - velocity[:-1, :, :]  # (帧数-1,24,3)，算出一帧内的加速度向量
    velocity = np.linalg.norm(velocity, axis=-1) # (帧数,24)

    """
    1、relative_velocity: (帧数,24)，各帧各节点的相对速度（单位是米/秒）
    """
    relative_velocity = np.concatenate([root_velocity, relative_velocity], axis=1) #(帧数,24,3)
    relative_velocity = np.linalg.norm(relative_velocity, axis=-1)  # (帧数,24)

    """
    2、acceleration: (帧数,24)，各帧各节点的速度（单位是米/秒平方）
    """
    first_frame = np.zeros_like(positions[0:1, ])  # (1,24,3)
    acceleration = np.concatenate([first_frame, acceleration], axis=0)  # (帧数,24,3)
    acceleration *= 60.0  # 60fps
    acceleration = np.linalg.norm(acceleration, axis=-1)  # (帧数,24)

    """
    frame_offsets: (帧数,23,3)，各帧各节点(除根节点，所以1:)相对于父节点的偏移
    bone_length: (23,)，各节点(除根节点)距离父节点的长度
    """
    frame_offsets = positions[:, 1:, :] - positions[:, parents[1:], :] # (帧数,23,3)

    bone_length = smpl_bvh_writer.SMPL_JOINTS_OFFSETS[smpl_bvh_writer.ROTATION_SEQ]
    bone_length = bone_length[1:, :] - bone_length[parents[1:], :]
    bone_length = np.linalg.norm(bone_length, axis=-1)

    """
    3、angular_velocity: (帧数,24)，各帧各节点的角速度标量（单位是弧度/秒）
    """
    angular_velocity = frame_offsets[1:, :, :] - frame_offsets[:-1, :, :] # (帧数-1,23,3)
    first_frame = np.zeros_like(frame_offsets[0:1:, ])  # (1,23,3)
    angular_velocity = np.concatenate([first_frame, angular_velocity], axis=0)  # (帧数,23,3)
    angular_velocity = np.linalg.norm(angular_velocity, axis=-1) # (帧数,23)
    angular_velocity *= 60.0  # 60fps，先模再60，更接近弧长
    angular_velocity = angular_velocity / bone_length # (帧数,23)
    first_bone = np.zeros_like(angular_velocity[:, 0:1])  # (帧数, 1)
    angular_velocity = np.concatenate([first_bone, angular_velocity], axis=1)  # (帧数,24)

    """
    omega: (帧数,24,3)，各帧各节点的物理含义的角速度矢量
    """
    omega = np.cross(frame_offsets[:-1, :, :], frame_offsets[1:, :, :]) # (帧数-1,23,3)，没有第一帧和根节点
    omega_norm = np.linalg.norm(omega, axis=-1) #(帧数-1,23)
    length_adjust = angular_velocity[1:, 1:] / omega_norm
    length_adjust = length_adjust[:, :, np.newaxis]
    omega = omega * length_adjust # (帧数-1,23,3)
    #temp = np.linalg.norm(omega, axis=-1)
    #temp = temp - angular_velocity[1:,1:] # 验证temp是0
    first_frame = np.zeros_like(omega[0:1:, ])  # (1,23,3)
    omega = np.concatenate([first_frame, omega], axis=0)  # (帧数,23,3)
    first_bone = np.zeros_like(omega[:, 0:1, :])  # (帧数, 1, 3)
    omega = np.concatenate([first_bone, omega], axis=1)  # (帧数,24,3)

    """
    4、angular_acceleration: (帧数,24)，各帧各节点的角速度（单位是弧度/秒平方）
    """
    angular_acceleration = omega[1:, :, :] - omega[:-1, :, :]  # (帧数-1,24,3)，算出一帧内的角加速度向量
    first_frame = np.zeros_like(angular_acceleration[0:1, ])  # (1,24,3)
    angular_acceleration = np.concatenate([first_frame, angular_acceleration], axis=0)  # (帧数,24,3)
    angular_acceleration *= 60.0  # 60fps
    angular_acceleration = np.linalg.norm(angular_acceleration, axis=-1)  # (帧数,24)

    """
    附加每帧的最大值、最小值
    """
    """
    main_nodes = [0, 2, 3, 6, 7, 13, 16, 17, 21, 22]
    #每帧速度的最大值
    max_v = np.amax(velocity[:, main_nodes], axis=1) # (帧数,)

    frames = max_v.shape[0]
    max_idx = np.zeros(frames, dtype=int)
    for i in range(max_v.shape[0]):
        if i == 0:
            continue
        vec = velocity[i,:]
        tup = np.where(vec == np.max(vec))
        max_idx[i] = tup[0]


    max_v = max_v[:, np.newaxis] # (帧数,1)
    min_v = np.average(velocity[:, main_nodes], axis=1)  # (帧数,)
    min_v = min_v[:, np.newaxis]  # (帧数,1)
    velocity = np.concatenate([velocity, max_v, min_v], axis=1) # (帧数,24+2)

    #每帧加速度的最大值
    max_a = np.amax(acceleration[:, main_nodes], axis=1)  # (帧数,)
    max_a = max_a[:, np.newaxis]  # (帧数,1)
    min_a = np.average(acceleration[:, main_nodes], axis=1)  # (帧数,)
    min_a = min_a[:, np.newaxis]  # (帧数,1)
    acceleration = np.concatenate([acceleration, max_a, min_a], axis=1) # (帧数,24+2)

    idxs = [2, 3, 6, 7, 13, 16, 17, 21, 22]

    # 每帧角速度的最大值
    max_av = np.amax(angular_velocity[:, main_nodes], axis=1)  # (帧数,)
    max_av = max_av[:, np.newaxis]  # (帧数,1)
    min_av = np.average(angular_velocity[:, main_nodes], axis=1)  # (帧数,)
    min_av = min_av[:, np.newaxis]  # (帧数,1)
    angular_velocity = np.concatenate([angular_velocity, max_av, min_av], axis=1)  # (帧数,24+2)

    # 每帧角加速度的最大值
    max_aa = np.amax(angular_acceleration[:, main_nodes], axis=1)  # (帧数,)
    max_aa = max_aa[:, np.newaxis]  # (帧数,1)
    min_aa = np.average(angular_acceleration[:, main_nodes], axis=1)  # (帧数,)
    min_aa = min_aa[:, np.newaxis]  # (帧数,1)
    angular_acceleration = np.concatenate([angular_acceleration, max_aa, min_aa], axis=1)  # (帧数,24+2)

    #每帧相对坐标绝对值之和
    root_offset = positions[:, 0:1, :] # (帧数,1,3)
    sum_offsets = np.concatenate([root_offset, frame_offsets], axis=1) # (帧数,24,3)
    sum_offsets = np.abs(sum_offsets)
    sum_offsets = np.sum(sum_offsets, axis = 2) # (帧数,24)
    sum_offsets = np.sum(sum_offsets, axis = 1) # (帧数,)
    sum_offsets = sum_offsets[:, np.newaxis]  # (帧数,1)
    """


    """
    5、音乐数据
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
        relative_velocity = relative_velocity[0:final_frames,]
        acceleration = acceleration[0:final_frames,]
        angular_velocity = angular_velocity[0:final_frames,]
        angular_acceleration = angular_acceleration[0:final_frames,]
        #sum_offsets = sum_offsets[0:final_frames,]
    else:
        music_feature = music_feature[0:final_frames,]

    """
    最终是(帧数, 24*5+3)，顺序是绝对速度24 + 相对速度24 + 加速度24 + 角速度24 + 角加速度24 + 音乐3
    """
    final_data = np.concatenate([velocity, relative_velocity, acceleration, angular_velocity, angular_acceleration, music_feature], axis=-1)

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