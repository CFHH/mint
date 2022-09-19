import numpy as np
import re

def load_bvh_motion(filename, scaled):
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

    if scaled:
        positions /= 100.0
        offsets /= 100.0
    rotations = rotations.reshape(frames, 24, 3)
    # 这俩直接给
    parents = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 11, 14, 15, 16, 17, 11, 19, 20, 21, 22]
    names = ['pelvis', 'l_hip', 'l_knee', 'l_ankle', 'l_foot', 'r_hip', 'r_knee', 'r_ankle', 'r_foot', 'spine1', 'spine2', 'spine3', 'neck', 'head', 'l_collar', 'l_shoulder', 'l_elbow', 'l_wrist', 'l_hand', 'r_collar', 'r_shoulder', 'r_elbow', 'r_wrist', 'r_hand']
    return names, parents, offsets, positions, rotations, frametime