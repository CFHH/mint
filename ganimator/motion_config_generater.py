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
import smpl_bvh_writer
from ganimator.tools.kinematics import ForwardKinematicsJoint
from ganimator.tools.Quaternions import Quaternions

FLAGS = flags.FLAGS
flags.DEFINE_string('motion_data_path', '../data/bvh_scale_split/', 'motion data path')
flags.DEFINE_boolean('scaled', True, 'if true, need / 100')

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
    return all_positions

def process(filename):
    names, parents, offsets, root_positions, rotations, frametime = load_bvh_motion(filename)
    parents = tuple(parents)
    offsets = torch.tensor(offsets, dtype=torch.float)
    fk = ForwardKinematicsJoint(parents, offsets)

    all_positions = joint_position(fk, root_positions, rotations)
    all_positions = all_positions.numpy()

    """
    起始帧根骨骼朝向
    起始帧各骨骼的位置（相对根骨骼，根骨骼xz归零时的位置，y是上下方向）、速度
    结束帧各骨骼的位置、速度
    结束帧根骨骼朝向
    整个动作的位移（结束帧相对起始帧的xz）
    """
    return 0

def main(_):
    process("../data/bvh_scale/gWA_sFM_cAll_d25_mWA4_ch05.bvh")
    return

    save_path = os.path.join(FLAGS.motion_data_path, "*.bvh")
    motion_files = glob.glob(os.path.join(FLAGS.motion_data_path, "*.bvh"))

    for file in tqdm.tqdm(motion_files):
        print("Process %s" % file)
        process(file)

if __name__ == '__main__':
    app.run(main)