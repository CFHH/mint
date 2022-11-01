import numpy as np
import matplotlib.pyplot as plt # https://matplotlib.org/stable/api/pyplot_summary.html
import os
import math

if __name__ == '__main__':
    """
    'pelvis', 'l_hip', 'l_knee', 'l_ankle', 'l_foot', 
    'r_hip', 'r_knee', 'r_ankle', 'r_foot', 'spine1',
    'spine2', 'spine3', 'neck', 'head', 'l_collar',
    'l_shoulder', 'l_elbow', 'l_wrist', 'l_hand', 'r_collar',
    'r_shoulder', 'r_elbow', 'r_wrist', 'r_hand', 'frame_max', 'frame_avg'
    选取：
    2、3、6、7、16、17、21、22、24（每帧最大值）、25（每帧最小值）
    """
    data = np.load('../data/velocity/gWA_sFM_cAll_d27_mWA5_ch20.npy')

    frames = data.shape[0]
    x = np.linspace(0, frames-1, frames)

    # 关键几个部位，在全序列中的索引
    show_ids = [2, 3, 6, 7, 16, 17, 21, 22, 24, 25]
    show_names = ['l_knee', 'l_ankle', 'r_knee', 'r_ankle', 'l_elbow', 'l_wrist', 'r_elbow', 'r_wrist', 'frame_max', 'frame_avg']

    startidx = 8 # 0、2、4、6，左脚、右脚、左手、右手、最大最小
    num = 2
    is_show = [True, False, True, False]

    fig, a = plt.subplots(num, 1)
    for i in range(num):
        # 速度
        if is_show[0]:
            velocity = data[:, 26*0 + show_ids[startidx + i]]
            a[i].plot(x, velocity, 'green', lw = 0.5)

        # 加速度
        if is_show[1]:
            acceleration = data[:, 26*1 + show_ids[startidx + i]] / 10.0
            a[i].plot(x, acceleration, 'purple', lw = 0.5)

        # 角速度
        if is_show[2]:
            angular_velocity = data[:, 26*2 + show_ids[startidx + i]]
            a[i].plot(x, angular_velocity, 'blue', lw = 0.5)

        # 角加速度
        if is_show[3]:
            angular_acceleration = data[:, 26*3 + show_ids[startidx + i]] / 10.0
            a[i].plot(x, angular_acceleration, 'yellow', lw=0.5)

        # strength
        #a[i].plot(x, data[:, 100 + 0], 'purple')

        # peak
        # a[i].plot(x, data[:, 100 + 1], 'purple')

        # beat
        a[i].plot(x, data[:, 26*4 + 2], 'red', lw = 0.5)

        #偏移绝对值和
        if False:
            offsets = data[:, 26 * 4 + 3] / 8.0
            a[i].plot(x, offsets, 'green', lw=0.5)

        a[i].set_title(show_names[startidx+i])
    plt.show()