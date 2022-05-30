import vedo
import trimesh
import torch
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy import linalg
import os



def eye(n, batch_shape):
    iden = np.zeros(np.concatenate([batch_shape, [n, n]]))
    iden[..., 0, 0] = 1.0
    iden[..., 1, 1] = 1.0
    iden[..., 2, 2] = 1.0
    return iden


def get_closest_rotmat(rotmats):
    """
    Finds the rotation matrix that is closest to the inputs in terms of the Frobenius norm. For each input matrix
    it computes the SVD as R = USV' and sets R_closest = UV'. Additionally, it is made sure that det(R_closest) == 1.
    Args:
        rotmats: np array of shape (..., 3, 3).
    Returns:
        A numpy array of the same shape as the inputs.
    """
    u, s, vh = np.linalg.svd(rotmats)
    r_closest = np.matmul(u, vh)

    # if the determinant of UV' is -1, we must flip the sign of the last column of u
    det = np.linalg.det(r_closest)  # (..., )
    iden = eye(3, det.shape)
    iden[..., 2, 2] = np.sign(det)
    r_closest = np.matmul(np.matmul(u, iden), vh)
    return r_closest


def recover_to_axis_angles(motion):
    # motion.shape = (1, 120, 220, 1)
    batch_size, seq_len, dim, _ = motion.shape
    motion = motion.reshape(batch_size, seq_len, dim)
    assert dim == 220
    pad_zeros = 1  # 本来是6
    transl = motion[:, :, pad_zeros:pad_zeros+3]
    rotmats = get_closest_rotmat(
        np.reshape(motion[:, :, pad_zeros+3:], (batch_size, seq_len, 24, 3, 3))
    )
    axis_angles = R.from_matrix(
        rotmats.reshape(-1, 3, 3)
    ).as_rotvec().reshape(batch_size, seq_len, 24, 3)
    return axis_angles, transl


def visualize(motion, smpl_model):
    smpl_poses, smpl_trans = recover_to_axis_angles(motion)
    smpl_poses = np.squeeze(smpl_poses, axis=0)  # (seq_len, 24, 3)
    smpl_trans = np.squeeze(smpl_trans, axis=0)  # (seq_len, 3)

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




if __name__ == "__main__":
    import glob
    import tqdm
    from smplx import SMPL

    smpl = SMPL(model_path="/home/ubuntuai/mint/others/smpl/", gender='MALE', batch_size=1)

    result_files = sorted(glob.glob("samples/*.npy"), key=os.path.getmtime)  #glob.glob("samples/*.npy")
    for result_file in tqdm.tqdm(result_files):
        print("Visual %s" % (result_file))
        result_motion = np.load(result_file)[None, ...]  # [1, 120, 220]
        visualize(result_motion, smpl)
