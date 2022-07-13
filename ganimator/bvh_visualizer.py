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

FLAGS = flags.FLAGS
flags.DEFINE_string('bvh_path', '../data/bvh/', 'Output path for the bvh files.')
flags.DEFINE_string('bvh_file', 'gBR_sBM_cAll_d04_mBR0_ch01', 'Output path for the bvh files.')

def visualize(smpl_model, smpl_trans, smpl_poses, frametime):
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

    i = 0
    while i < keypoints3d.shape[0]:
        keypoints3d_i = keypoints3d[i]
        vertices_i = vertices[i]
        i += 1
        mesh = trimesh.Trimesh(vertices_i, smpl_model.faces)
        mesh.visual.face_colors = [200, 200, 250, 100]
        pts = vedo.Points(keypoints3d_i, r=20)
        plotter = vedo.show(world, mesh, interactive=0)
        if plotter.escaped:
            break  # if ESC
        time.sleep(0.005)

    vedo.interactive().close()

def load_bvh_motion(filename):
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
    return positions, rotations, frametime

def main(_):
    smpl_model = SMPL(model_path="../others/smpl/", gender='MALE', batch_size=1)
    filename = os.path.join(FLAGS.bvh_path, '%s.bvh' % FLAGS.bvh_file)
    positions, rotations, frametime = load_bvh_motion(filename)
    visualize(smpl_model, positions, rotations, frametime)

if __name__ == '__main__':
  app.run(main)
