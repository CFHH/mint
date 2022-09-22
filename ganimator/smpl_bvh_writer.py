import numpy as np

# 这个顺序是aist++数据集中的顺序，也是SMPL的默认顺序，不能改
SMPL_JOINTS_NAMES = [
    "pelvis",       #0
    "l_hip",        #1
    "r_hip",        #2
    "spine1",       #3
    "l_knee",       #4
    "r_knee",       #5
    "spine2",       #6
    "l_ankle",      #7
    "r_ankle",      #8
    "spine3",       #9
    "l_foot",       #10
    "r_foot",       #11
    "neck",         #12
    "l_collar",     #13
    "r_collar",     #14
    "head",         #15
    "l_shoulder",   #16
    "r_shoulder",   #17
    "l_elbow",      #18
    "r_elbow",      #19
    "l_wrist",      #20
    "r_wrist",      #21
    "l_hand",       #22 四根手指与手掌的连接处
    "r_hand",       #23
]

# SMPL_JOINTS_NAMES中各节点的父节点
SMPL_JOINTS_PARENTS = [
    -1, 0, 0, 0, 1,
    2, 3, 4, 5, 6,
    7, 8, 9, 9, 9,
    12, 13, 14, 16, 17,
    18, 19, 20, 21,
]

# 这是T-pos下各节点的世界坐标，不是相对父节点的坐标
SMPL_JOINTS_OFFSETS = np.array(
    [
        [-8.76308970e-04, -2.11418723e-01, 2.78211200e-02],     #0
        [7.04848876e-02, -3.01002533e-01, 1.97749280e-02],      #1
        [-6.98883278e-02, -3.00379160e-01, 2.30254335e-02],     #2
        [-3.38451650e-03, -1.08161861e-01, 5.63597909e-03],     #3
        [1.01153808e-01, -6.65211904e-01, 1.30860155e-02],      #4
        [-1.06040718e-01, -6.71029623e-01, 1.38401121e-02],     #5
        [1.96440985e-04, 1.94957852e-02, 3.92296547e-03],       #6
        [8.95999143e-02, -1.04856032e00, -3.04155922e-02],      #7
        [-9.20120818e-02, -1.05466743e00, -2.80514913e-02],     #8
        [2.22362284e-03, 6.85680141e-02, 3.17901760e-02],       #9
        [1.12937580e-01, -1.10320516e00, 8.39545265e-02],       #10
        [-1.14055299e-01, -1.10107698e00, 8.98482216e-02],      #11
        [2.60992373e-04, 2.76811197e-01, -1.79753042e-02],      #12
        [7.75218998e-02, 1.86348444e-01, -5.08464100e-03],      #13
        [-7.48091986e-02, 1.84174211e-01, -1.00204779e-02],     #14
        [3.77815350e-03, 3.39133394e-01, 3.22299558e-02],       #15
        [1.62839013e-01, 2.18087461e-01, -1.23774789e-02],      #16
        [-1.64012068e-01, 2.16959041e-01, -1.98226746e-02],     #17
        [4.14086325e-01, 2.06120683e-01, -3.98959248e-02],      #18
        [-4.10001734e-01, 2.03806676e-01, -3.99843890e-02],     #19
        [6.52105424e-01, 2.15127546e-01, -3.98521818e-02],      #20
        [-6.55178550e-01, 2.12428626e-01, -4.35159074e-02],     #21
        [7.31773168e-01, 2.05445019e-01, -5.30577698e-02],      #22
        [-7.35578759e-01, 2.05180646e-01, -5.39352281e-02],     #23
    ]
)

"""
这是转成bvh文件后，各骨骼节点的出场次序，
['pelvis', 'l_hip', 'l_knee', 'l_ankle', 'l_foot', 'r_hip', 'r_knee', 'r_ankle', 'r_foot', 'spine1', 
 'spine2', 'spine3', 'neck', 'head', 'l_collar', 'l_shoulder', 'l_elbow', 'l_wrist', 'l_hand', 'r_collar', 
 'r_shoulder', 'r_elbow', 'r_wrist', 'r_hand']
ROTATION_SEQ把smpl顺序变成bvh顺序
ROTATION_SEQ_INV把bvh顺序变成smpl顺序
"""
ROTATION_SEQ = [0, 1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12, 15, 13, 16, 18, 20, 22, 14, 17, 19, 21, 23]
ROTATION_SEQ_INV = [0, 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12, 14, 19, 13, 15, 20, 16, 21, 17, 22, 18, 23]

def write_smpl_bvh(filename, names, parent, offset, xyz_ratation_order, positions, rotations, frametime, scale100=False):
    file = open(filename, 'w')
    joints_num = len(names)
    xyz_ratation_order = xyz_ratation_order.upper()
    frames = rotations.shape[0]

    file_string = 'HIERARCHY\n'
    seq = [] # 最终就是ROTATION_SEQ

    def write_static(idx, prefix):
        nonlocal names, parent, offset, xyz_ratation_order, positions, rotations, frames, file_string, seq
        seq.append(idx)
        if idx == 0:
            name_label = 'ROOT ' + names[idx]
            channel_label = 'CHANNELS 6 Xposition Yposition Zposition {}rotation {}rotation {}rotation'.format(*xyz_ratation_order)
        else:
            name_label = 'JOINT ' + names[idx]
            channel_label = 'CHANNELS 3 {}rotation {}rotation {}rotation'.format(*xyz_ratation_order)
        if scale100:
            offset_label = 'OFFSET %.6f %.6f %.6f' % (offset[idx][0] * 100.0, offset[idx][1] * 100.0, offset[idx][2] * 100.0)
        else:
            offset_label = 'OFFSET %.6f %.6f %.6f' % (offset[idx][0], offset[idx][1], offset[idx][2])

        file_string += prefix + name_label + '\n'
        file_string += prefix + '{\n'
        file_string += prefix + '\t' + offset_label + '\n'
        file_string += prefix + '\t' + channel_label + '\n'

        has_child = False
        for y in range(idx+1, joints_num):
            if parent[y] == idx:
                has_child = True
                write_static(y, prefix + '\t')
        if not has_child:
            file_string += prefix + '\t' + 'End Site\n'
            file_string += prefix + '\t' + '{\n'
            file_string += prefix + '\t\t' + 'OFFSET 0 0 0\n'
            file_string += prefix + '\t' + '}\n'

        file_string += prefix + '}\n'

    write_static(0, '')

    file_string += 'MOTION\n' + 'Frames: {}\n'.format(frames) + 'Frame Time: %.8f\n' % frametime
    for i in range(frames):
        if scale100:
            file_string += '%.6f %.6f %.6f ' % (positions[i][0] * 100.0, positions[i][1] * 100.0, positions[i][2] * 100.0)
        else:
            file_string += '%.6f %.6f %.6f ' % (positions[i][0], positions[i][1], positions[i][2])
        for j in range(joints_num):
            idx = seq[j]
            file_string += '%.6f %.6f %.6f ' % (rotations[i][idx][0], rotations[i][idx][1], rotations[i][idx][2])
        file_string += '\n'

    file.write(file_string)

    """
    seq_string = '['
    for i in range(len(seq)):
        if i != len(seq) -1:
            seq_string += '%d, ' % seq[i]
        else:
            seq_string += '%d' % seq[i]
    seq_string += ']'
    """
    return

def save_motion_as_bvh(filename, positions, ratations, frametime, scale100=True):
    smpl_offsets = np.zeros([24, 3])
    smpl_offsets[0] = [0.0, 0.0, 0.0]
    for idx, pid in enumerate(SMPL_JOINTS_PARENTS[1:]):
        smpl_offsets[idx + 1] = SMPL_JOINTS_OFFSETS[idx + 1] - SMPL_JOINTS_OFFSETS[pid]

    if scale100:
        #为了在bvhacker里显示，位移必须扩大100倍（就当作是单位从米变成厘米）
        #而让程序知道放大； 100倍，就必须放根节点的offset的Y大于10
        #所以，在此处初始Y+1，后面动作的Y-1
        smpl_offsets[0][1] = 1.0
        frames, _ = positions.shape
        for i in range(frames):
            positions[i][1] = positions[i][1] - 1.0 # 这会改变原始数据

    write_smpl_bvh(filename, SMPL_JOINTS_NAMES, SMPL_JOINTS_PARENTS, smpl_offsets, 'xyz', positions, ratations, frametime, scale100=scale100)

    if scale100:
        frames, _ = positions.shape
        for i in range(frames):
            positions[i][1] = positions[i][1] + 1.0  # 改回来

def test():
    dummy_frames = 10
    base_position = [0.0, 0.9, 0.0]
    positions = np.zeros([dummy_frames, 3])
    for i in range(dummy_frames):
        positions[i] = base_position
    ratations = np.zeros([dummy_frames, 24, 3])
    # 实测得，旋转角度的含义是绕XYZ轴，按右手螺旋旋转，是intrinsic rotations
    # intrinsic rotations的含义就是旋转轴是跟随物体的，见
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.as_euler.html#r72d546869407-1
    # X：人物自身的右脚朝左脚；Y：指向人物头顶方向；Z：右手系根据XY确定Z
    # 先X再Y（x=75，y=90），先Y再Z（y=75，z=90），先X再Z（x=90，z=90）
    ratations[:, 0, 0] = 75
    ratations[:, 0, 1] = 0
    ratations[:, 0, 2] = 90
    save_motion_as_bvh('./my.bvh', positions, ratations, 0.033333333, True)

if __name__ == '__main__':
    test()