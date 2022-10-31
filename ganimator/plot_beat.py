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
    'r_shoulder', 'r_elbow', 'r_wrist', 'r_hand'
    选取：
    2、3、6、7、16、17、21、22
    """
    data = np.load('../data/velocity/gWA_sFM_cAll_d25_mWA4_ch05.npy')

    frames = data.shape[0]
    x = np.linspace(0, frames-1, frames)

    y_l_knee = data[:, 2+24]
    y_l_ankle = data[:, 3+24]
    y_r_knee = data[:, 6+24]
    y_r_ankle = data[:, 7+24]
    y_l_elbow = data[:, 16+24]
    y_l_wrist = data[:, 17+24]
    y_r_elbow = data[:, 21+24]
    y_r_wrist = data[:, 22+24]

    y_strength = data[:, 0+48]
    y_peak_onehot = data[:, 1+48]
    y_beat_onehot = data[:, 2+48]


    show_ids = [2, 3, 6, 7, 16, 17, 21, 22]
    show_names = ['l_knee', 'l_ankle', 'r_knee', 'r_ankle', 'l_elbow', 'l_wrist', 'r_elbow', 'r_wrist']
    num = len(show_ids)

    startidx = 4
    fig, a = plt.subplots(2, 1)
    for i in range(2):
        # 角速度
        a[i].plot(x, data[:, 24 + show_ids[startidx+i]], 'blue')
        # 速度
        a[i].plot(x, data[:, 0 + show_ids[startidx+i]], 'green')
        # strength
        a[i].plot(x, data[:, 48 + 0], 'yellow')
        # beat
        a[i].plot(x, data[:, 48 + 2], 'red')
        a[i].set_title(show_names[startidx+i])
    plt.show()