from absl import app
from absl import flags
from absl import logging
import re
import numpy as np
import math
import torch
import time
import os
from scipy.spatial.transform import Rotation as R
from scipy import linalg
import glob
import tqdm
from smplx import SMPL
import vedo
import trimesh
import smpl_bvh_writer

# python bvh_visualizer.py --bvh_path=../data/bvh --bvh_file=gWA_sFM_cAll_d25_mWA4_ch05
# python bvh_visualizer.py --bvh_path=../../ganimator/results/gWA_sFM_cAll_d25_mWA4_ch05/bvh --bvh_file=result_fixed

FLAGS = flags.FLAGS
flags.DEFINE_string('bvh_path', '../data/bvh/', 'file path')
flags.DEFINE_string('bvh_file', 'gWA_sFM_cAll_d25_mWA4_ch05', 'file name')
flags.DEFINE_integer('bpm', 120, 'bpm')
flags.DEFINE_float('ignoretime', 0.0, 'ignore time')

def is_on_beat(motion_time, bpm):
    motion_time -= FLAGS.ignoretime
    beat_time = 60.0 / float(bpm)
    tolerance = 0.03 #30毫秒，帧率不高
    delta_time = motion_time - beat_time * float(int((motion_time + tolerance) / beat_time))
    if math.fabs(delta_time) < tolerance:
        print(motion_time)
        return True
    else:
        return False

def visualize(smpl_model, smpl_trans, smpl_poses, frametime, bpm=120):
    #position: smpl_trans.shape = (frames, 3)
    #rotation: smpl_poses.shape = (frames, 24, 3)

    smpl_output = smpl_model.forward(
        global_orient=torch.from_numpy(smpl_poses[:, 0:1]).float(),
        body_pose=torch.from_numpy(smpl_poses[:, 1:]).float(),
        transl=torch.from_numpy(smpl_trans).float(),
    )
    keypoints3d = smpl_output.joints.detach().numpy()   # (seq_len, 24, 3)
    vertices = smpl_output.vertices.detach().numpy()

    bbox_center = (
        keypoints3d.reshape(-1, 3).max(axis=0)
        + keypoints3d.reshape(-1, 3).min(axis=0)
    ) / 2.0
    bbox_size = (
        keypoints3d.reshape(-1, 3).max(axis=0)
        - keypoints3d.reshape(-1, 3).min(axis=0)
    )
    world = vedo.Box(bbox_center, bbox_size[0], bbox_size[1], bbox_size[2]).wireframe()
    vedo.show(world, axes=True, viewup="y", interactive=0)

    start_time = time.time() #类似这样的一个数字：1657703177.133365，单位上秒
    last_time = start_time
    while True:
        cur_time = time.time()
        if cur_time < last_time:
            cur_time = last_time
        else:
            last_time = cur_time
        delta_time = cur_time - start_time
        frame_index = int(delta_time / frametime)
        if frame_index >= keypoints3d.shape[0]:
            break

        keypoints3d_i = keypoints3d[frame_index]
        vertices_i = vertices[frame_index]

        mesh = trimesh.Trimesh(vertices_i, smpl_model.faces)
        if is_on_beat(delta_time, bpm):
            mesh.visual.face_colors = [200, 50, 50, 100]
        else:
            mesh.visual.face_colors = [200, 200, 250, 100]
        pts = vedo.Points(keypoints3d_i, r=20)
        plotter = vedo.show(world, mesh, interactive=0)
        if plotter.escaped:
            break  # if ESC

    vedo.interactive().close()

def load_bvh_motion(filename):
    """
    seq = smpl_bvh_writer.ROTATION_SEQ
    inv_seq = np.zeros(len(seq), dtype=np.int32)
    dic = {}
    for i in range(len(seq)):
        dic[seq[i]] = i
    for i in range(len(seq)):
        inv_seq[i] = dic[i]
    """

    cur_step = 0
    f = open(filename, "r")
    for line in f:
        if cur_step == 0:
            if "MOTION" in line:
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
            position /= 100.0
            position[1] += 1.0

            rotation = data_block[3:]
            rotation *= (math.pi / 180)

            positions[i] = position
            rotations[i] = rotation
            i += 1

    rotations = rotations.reshape(frames, 24, 3)
    rotations = rotations[:, smpl_bvh_writer.ROTATION_SEQ_INV, :]
    return positions, rotations, frametime

def main(_):
    smpl_model = SMPL(model_path="../others/smpl/", gender='MALE', batch_size=1)
    filename = os.path.join(FLAGS.bvh_path, '%s.bvh' % FLAGS.bvh_file)
    positions, rotations, frametime = load_bvh_motion(filename)
    visualize(smpl_model, positions, rotations, frametime, bpm=FLAGS.bpm)

if __name__ == '__main__':
  app.run(main)
