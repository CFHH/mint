from absl import app
from absl import flags
from absl import logging
import os
import numpy as np
import math
import random
import json
import torch
import bvh_loader
import smpl_bvh_writer
from tools.Quaternions import Quaternions
from tools.transforms import quat2euler

FLAGS = flags.FLAGS
flags.DEFINE_string('motion_data_path', '../data/bvh_scale_split/', 'motion data path')
flags.DEFINE_boolean('scaled', True, '动作是否已经放大了100倍')
flags.DEFINE_string('motion_json_path', '../data/bvh_scale_split/', 'motion data for synthesis')
flags.DEFINE_integer('target_number', 30, 'target motion segment number')

flags.DEFINE_string('save_path', './', '保存目录')
flags.DEFINE_string('save_name', 'test', '保存目录')
flags.DEFINE_boolean('save_scaled', True, 'scale position and offset')

def load_motion_config():
    json_data = None
    json_file = os.path.join(FLAGS.motion_json_path, "synthesis_motion.dat")
    if os.path.isfile(json_file):
        with open(json_file, 'r') as f:
            json_str = f.read()
            json_data = json.loads(json_str)
    return json_data

def calc_transition_cost(json_data, key1, key2, accumulated_offset):
    motion1 = json_data[key1]
    motion2 = json_data[key2]

    # (24,3)
    position_diff = np.array(motion2['start_position']) - np.array(motion1['end_position'])
    position_diff = np.abs(position_diff).sum()
    # (3,)
    rotation_diff = np.array(motion2['start_rotation']) - np.array(motion1['end_rotation'])
    rotation_diff = np.abs(rotation_diff).sum()
    # (24,3)
    velocity_diff = np.array(motion2['start_velocity']) - np.array(motion1['end_velocity'])
    velocity_diff = np.abs(velocity_diff).sum()
    # (3,)
    root_offset = np.array(motion2['root_offset'])
    offset_length = np.linalg.norm(accumulated_offset + root_offset)

    if position_diff > 700 or rotation_diff > 45 or velocity_diff > 200 or offset_length > 500:
        cost = 9999.0 #
    else:
        # 各骨骼位置差异（单位米）的和 + 各骨骼旋转差异（单位弧度）的和 + 各骨骼速度差异（米/秒）的和
        cost = position_diff / 100.0 + rotation_diff * math.pi / 180.0 + velocity_diff / 100.0 * 30.0 + offset_length

    return cost

def synthesiser_as_bvh(json_data, motion_name_seq, save_file, transition_time = 0.2):
    print(motion_name_seq)
    FrameTime = 0.01666667
    transition_frames = round(transition_time / FrameTime)
    pre_frame_num = int((transition_frames + 1) / 2)
    cur_frame_num = int(transition_frames - pre_frame_num)
    pre_t = []
    cur_t = []
    for i in range(pre_frame_num):
        pre_t.append(0.5 * (i+1) / pre_frame_num)
    for i in range(cur_frame_num):
        cur_t.append(0.5 - 0.5 * i / cur_frame_num)

    final_positions = []
    final_rotations = []
    accumulated_offset = np.array([0.0, 0.0, 0.0])
    total_frames = 0

    num = len(motion_name_seq)
    for seq_index in range(num):
        motion_name = motion_name_seq[seq_index]
        motion_data = json_data[motion_name]
        filename = os.path.join(FLAGS.motion_data_path, motion_name)
        """
        positions: (帧数, 3)，np.array，各帧根骨骼的位移（单位是米）
        rotations: (帧数, 24, 3)，np.array，各帧各骨骼节点的欧拉角（单位是角度）
        """
        _, _, offsets, positions, rotations, _ = bvh_loader.load_bvh_motion(filename, FLAGS.scaled)

        if seq_index == 0:
            accumulated_offset[1] += positions[0, 1] + offsets[0, 1]

        # 一、根骨骼位置
        root_adjust = accumulated_offset - positions[0,]
        positions += root_adjust
        accumulated_offset = positions[-1,]
        if seq_index != num - 1:
            positions = positions[0:-1, ]  # 舍弃最后一帧
        final_positions.append(positions)
        total_frames += positions.shape[0]

        # 二、各骨骼欧拉角
        if seq_index == 0:
            final_rotations.append(rotations)
        else:
            """
            各骨骼节点的欧拉角插值
            对上个动作的末尾几帧和当前动作的开头几帧，做插值
            如果fps=60，那么2秒的动作共121帧数据，2个2秒的动作合并后应该是241帧，而不是242帧，省略上一动作的最后一帧
            P P P P P P P P P P
                              C C C C C C C C C C
            """
            pre_rotations = final_rotations.pop() # 弹出上一帧

            pre_last_rotation = pre_rotations[-1,]
            pre_rotations = pre_rotations[0:-1,] #舍弃最后一帧
            pre_last_quat = Quaternions.from_euler(np.radians(pre_last_rotation), world=False)

            cur_first_rotation = rotations[0,]
            cur_first_quat = Quaternions.from_euler(np.radians(cur_first_rotation), world=False)

            for i in range(pre_frame_num):
                pre_quat = Quaternions.from_euler(np.radians(pre_rotations[-pre_frame_num+i]), world=False)
                slerp_quat = Quaternions.slerp(pre_quat, cur_first_quat, pre_t[i])
                slerp_euler = np.degrees(slerp_quat.euler())
                pre_rotations[-pre_frame_num + i] = slerp_euler

            for i in range(cur_frame_num):
                cur_quat = Quaternions.from_euler(np.radians(rotations[i]), world=False)
                slerp_quat = Quaternions.slerp(cur_quat, pre_last_quat, cur_t[i])
                slerp_euler = np.degrees(slerp_quat.euler())
                rotations[i] = slerp_euler

            final_rotations.append(pre_rotations) # 重新压入上一帧
            final_rotations.append(rotations)

    merged_positions = np.zeros([total_frames,3])
    merged_rotations = np.zeros([total_frames,24, 3])
    start = 0
    for i in range(num):
        frames = final_positions[i].shape[0]
        assert final_rotations[i].shape[0] == frames
        end = start + frames
        merged_positions[start:end] = final_positions[i]
        merged_rotations[start:end] = final_rotations[i]
        start = end
    smpl_bvh_writer.save_motion_as_bvh(save_file, merged_positions, merged_rotations, FrameTime, scale100=FLAGS.save_scaled, rotation_order='bvh')


def main(_):
    json_data = load_motion_config()
    motion_names = []
    for _, name in enumerate(json_data):
        motion_names.append(name)
    motion_num = len(motion_names)

    """ test
    while True:
        index1 = random.randrange(0, motion_num)
        index2 = random.randrange(0, motion_num)
        position_diff, rotation_diff, velocity_diff, root_offset = calc_transition_cost(json_data, motions[index1], motions[index2])
        print("%f, %f, %f" % (position_diff, rotation_diff, velocity_diff))
    return
    """

    motion_seq = []
    accumulated_offset = np.array([0.0, 0.0, 0.0])

    while True:
        length = len(motion_seq)
        if length >= FLAGS.target_number:
            break
        if length == 0:
            index = random.randrange(0, motion_num)
            motion_seq.append(index)
            accumulated_offset += np.array(json_data[motion_names[index]]['root_offset'])
        else:
            pre_index = motion_seq[-1]
            cur_index = -1
            cur_cost = 999.0

            for i in range(100): # 直接生成100个不重复的 random.sample(range(0, motion_num), 100)
                index = random.randrange(0, motion_num)
                if index in motion_seq or index == cur_index:
                    continue
                cost = calc_transition_cost(json_data, motion_names[pre_index], motion_names[index], accumulated_offset)
                if cost >= cur_cost:
                    continue
                cur_index = index
                cur_cost = cost

            if cur_index != -1:
                print("Step forward")
                # 找到了
                motion_seq.append(cur_index)
                accumulated_offset += np.array(json_data[motion_names[cur_index]]['root_offset'])
                if length + 1 == FLAGS.target_number:
                    print("Complete")
                    break
            else:
                print("Step backward")
                motion_seq.pop()
                accumulated_offset -= np.array(json_data[motion_names[pre_index]]['root_offset'])

    print("accumulated_offset = (%f, %f, %f)" % (accumulated_offset[0], accumulated_offset[1], accumulated_offset[2]))
    motion_name_seq = []
    for i in range(len(motion_seq)):
        motion_name_seq.append(motion_names[motion_seq[i]])

    save_file = os.path.join(FLAGS.save_path, '%s.bvh' % FLAGS.save_name)
    synthesiser_as_bvh(json_data, motion_name_seq, save_file)

if __name__ == '__main__':
    app.run(main)