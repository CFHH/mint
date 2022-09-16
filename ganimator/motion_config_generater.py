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
import smpl_bvh_writer
from tools.kinematics import ForwardKinematicsJoint
from tools.Quaternions import Quaternions

FLAGS = flags.FLAGS
flags.DEFINE_string('motion_data_path', '../data/bvh_scale_split/', 'motion data path')
flags.DEFINE_boolean('scaled', True, 'if true, need / 100')
flags.DEFINE_string('motion_json_path', '../data/bvh_scale_split/', 'motion data for synthesis')

def load_bvh_motion(filename):
    cur_step = 0
    f = open(filename, "r")
    offsets = np.array([]).reshape((0, 3))
    read_offset = False
    for line in f:
        if cur_step == 0:
            if read_offset:
                offmatch = re.match(r"\s*OFFSET\s+([\-\d\.e]+)\s+([\-\d\.e]+)\s+([\-\d\.e]+)", line)
                if offmatch:
                    read_offset = False
                    offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
                    offsets[-1] = np.array([list(map(float, offmatch.groups()))])
            elif "ROOT" in line or "JOINT" in line:
                read_offset = True
            elif "MOTION" in line:
                cur_step = 1
            continue
        if cur_step == 1:
            cur_step = 2
            fmatch = re.match("\s*Frames:\s+(\d+)", line)
            frames = int(fmatch.group(1))
            positions = np.zeros((frames, 3), dtype=np.float32)
            rotations = np.zeros((frames, 72), dtype=np.float32)
            continue
        if cur_step == 2:
            cur_step = 3
            i = 0
            fmatch = re.match("\s*Frame Time:\s+([\d\.]+)", line)
            frametime = float(fmatch.group(1))
            continue
        if cur_step == 3:
            dmatch = line.strip().split()
            data_block = np.array(list(map(float, dmatch)))
            assert len(data_block) == 75
            position = data_block[0:3]
            rotation = data_block[3:]
            positions[i] = position
            rotations[i] = rotation
            i += 1

    if FLAGS.scaled:
        positions /= 100.0
        offsets /= 100.0
    rotations = rotations.reshape(frames, 24, 3)
    # 这俩直接给
    parents = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 11, 14, 15, 16, 17, 11, 19, 20, 21, 22]
    names = ['pelvis', 'l_hip', 'l_knee', 'l_ankle', 'l_foot', 'r_hip', 'r_knee', 'r_ankle', 'r_foot', 'spine1', 'spine2', 'spine3', 'neck', 'head', 'l_collar', 'l_shoulder', 'l_elbow', 'l_wrist', 'l_hand', 'r_collar', 'r_shoulder', 'r_elbow', 'r_wrist', 'r_hand']
    return names, parents, offsets, positions, rotations, frametime

def joint_position(fk, root_positions, rotations):
    root_positions = torch.tensor(root_positions, dtype=torch.float)
    rotations = Quaternions.from_euler(np.radians(rotations)).qs
    rotations = torch.tensor(rotations, dtype=torch.float)
    all_positions = fk.forward(rotations, root_positions) # 计算各骨骼节点的世界坐标位置(frame, 24, 3)
    return all_positions.numpy()

def process(filename):
    """
    names         : (24,)，各骨骼节点的名字，24按bvh中出场次序（下同）
    parents       : (24,)，各骨骼节点的父节点索引
    offsets       : (24,3)，各骨骼节点在初始状态下的对父节点的位移（单位是米），也就是bvh中的OFFSET
    root_positions: (帧数,3)，各帧根骨骼的位移（单位是米）
    rotations     : (帧数,24,3)，各帧各骨骼节点的欧拉角（单位是角度）
    以上数字数据的类型都是numpy数组
    """
    names, parents, offsets, root_positions, rotations, frametime = load_bvh_motion(filename)
    parents_temp = tuple(parents)
    offsets_temp = torch.tensor(offsets, dtype=torch.float)
    fk = ForwardKinematicsJoint(parents_temp, offsets_temp)

    """
    positions: (帧数,24,3)，各帧各节点的世界坐标
    """
    positions = joint_position(fk, root_positions, rotations)

    """
    start_rotation  : (3,)，起始帧，根骨骼的欧拉角（单位是角度）【是否只需要Y的旋转】
    start_position_r: (24,3)，起始帧，各骨骼节点相对根骨骼的位移（单位是厘米）
    start_velocity  : (24,3)，起始帧，各骨骼节点的速度（在1/30秒内的位移，单位是厘米）
    end_rotation    : (3,)，结束帧
    end_position_r  : (24,3)，结束帧
    end_velocity    : (24,3)，结束帧
    root_offset     : (3,)，整个动作期间，根骨骼的偏移（单位是厘米）【是否不需要考虑Y】
    
    起始帧各骨骼的位置（相对根骨骼，根骨骼xz归零时的位置，y是上下方向）、速度
    结束帧各骨骼的位置、速度
    结束帧根骨骼朝向
    整个动作的位移（结束帧相对起始帧的xz）
    """
    start_rotation = rotations[0,0,:]
    start_position = positions[0,]
    start_position_r = start_position - start_position[0,]
    start_velocity = positions[2,]-positions[0,]
    end_rotation = rotations[-1,0,:]
    end_position = positions[-1,]
    end_position_r = end_position - end_position[0,]
    end_velocity = positions[-1,] - positions[-3,]
    root_offset = root_positions[-1,] - root_positions[0,]
    #米到厘米
    start_position_r = start_position_r * 100
    start_velocity = start_velocity * 100
    end_position_r = end_position_r * 100
    end_velocity = end_velocity * 100
    root_offset = root_offset * 100

    motion_data = {'start_rotation': start_rotation.tolist(),
                 'start_position': start_position_r.tolist(),
                 'start_velocity': start_velocity.tolist(),
                 'end_rotation': end_rotation.tolist(),
                 'end_position': end_position_r.tolist(),
                 'end_velocity': end_velocity.tolist(),
                 'root_offset': root_offset.tolist(),
                 }
    """test
    json_data = {}
    json_data['motion_name'] = motion_data
    json_str = json.dumps(json_data)

    json_data2 = json.loads(json_str)
    motion_data2 = json_data2['motion_name']
    json_start_position = motion_data2['start_position']
    json_start_position_np = np.array(json_start_position)
    diff = json_start_position_np - start_position_r
    """

    return motion_data

def main(_):
    # 调试用
    #process("../data/bvh_scale/gWA_sFM_cAll_d25_mWA4_ch05.bvh")

    json_data = {}
    save_path = os.path.join(FLAGS.motion_data_path, "*.bvh")
    motion_files = glob.glob(os.path.join(FLAGS.motion_data_path, "*.bvh"))
    for file in tqdm.tqdm(motion_files):
        print("Process %s" % file)
        motion_data = process(file)
        key = file.split("\\")[-1] # file类似这样： '../data/bvh_scale_split\\gBR_sBM_cAll_d04_mBR0_ch01_0.bvh'
        json_data[key] = motion_data

    json_str = json.dumps(json_data)
    json_file = os.path.join(FLAGS.motion_json_path, "synthesis_motion.dat")
    with open(json_file, 'w') as f:
        f.write(json_str)

    return

if __name__ == '__main__':
    app.run(main)