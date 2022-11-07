import numpy as np
import matplotlib.pyplot as plt # https://matplotlib.org/stable/api/pyplot_summary.html
import os
import math
from scipy import signal

def filter(x):
    print('filtering data...')
    b, a = signal.butter(8, [0.01, 0.4], 'bandpass')
    fv = signal.filtfilt(b, a, x)   #data为要过滤的信号
    fv = fv - fv.min()
    print('done')
    return fv

def avg_filter(dat, kernel_size = 5):
    length = dat.shape[0]
    neighbor = kernel_size // 2
    fdat = np.zeros(length, dtype=float)
    for i in range(length):
        from_idx = max(i - neighbor, 0)
        to_idx = min(i + neighbor, length-1)
        if to_idx != length-1:
            fdat[i] = np.average(dat[from_idx:to_idx+1])
        else:
            fdat[i] = np.average(dat[from_idx:])
    return fdat


def norm_probability_density_function(x):
    np.exp(-x*x/2)
    return x



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
    data = np.load('../data/velocity/gBR_sBM_cAll_d04_mBR1_ch10.npy')
    frames = data.shape[0]
    velocity = data[:, 0:24]
    relative_velocity = data[:, 24:48]
    acceleration = data[:, 48:72]
    angular_velocity = data[:, 72:96]
    angular_acceleration = data[:, 96:120]
    strength = data[:, 120+0]
    peak_onehot = data[:, 120+1]
    beat_onehot = data[:, 120+2]

    #速度，相对于自己平均值的百分比
    bone_avg = np.average(relative_velocity, axis=0)
    bone_avg_pct = relative_velocity / bone_avg

    """
    额外算点数据
    """
    #动量
    mass = [100.0, 20.0, 10.0, 2.0, 1.0,
            20.0, 10.0, 8.0, 1.0, 60.0,
            50.0, 40.0, 10.0, 8.0, 15.0,
            10.0, 5.0, 2.0, 1.0, 15.0,
            10.0, 5.0, 2.0, 1.0]
    momentum = relative_velocity * mass
    momentum = np.sum(momentum, axis = 1)
    momentum /= 100.0

    energy = relative_velocity * relative_velocity * mass
    energy = np.sum(energy, axis=1)
    energy /= 100

    #速度最大的N个的平均
    value = velocity.copy() * 3.0
    n = 5
    max_n = np.zeros((frames,n), dtype=float)
    for i in range(frames):
        if i == 0:
            continue
        vec = value[i, :]
        for nn in range(n):
            tup = np.where(vec == np.max(vec))
            max_n[i,nn] = vec[tup[0]]
            vec[tup[0]] = 0.0
    max_n_avg = np.average(max_n, axis=1)


    #注意点
    max_idx = np.zeros(frames, dtype=int)
    last_idx = -1
    for i in range(frames):
        if i == 0:
            continue
        vec = velocity[i,:]
        tup = np.where(vec == np.max(vec))
        if last_idx in tup:
            max_idx[i] = last_idx
        else:
            max_idx[i] = tup[0]
            last_idx = max_idx[i]

    cur_id = -1
    cur_cnt = 0
    i = 0
    while True:
        if i >= frames:
            break
        idx = max_idx[i]
        if idx == cur_id:
            cur_cnt += 1
            i += 1
            continue
        else:
            if cur_cnt > 15:
                #extra_cnt = min(cur_cnt // 10, 20, frames-i)
                extra_cnt = min(6, frames-i)
                max_idx[i:i+extra_cnt] = cur_id
                cur_cnt = extra_cnt # 重新计数
                i += extra_cnt
            else:
                cur_id = idx
                cur_cnt = 1
                i += 1

    attention_max = np.zeros(frames, dtype=float)
    for i in range(frames):
        attention_max[i] = velocity[i, max_idx[i]]


    #每帧速度的最大值
    main_nodes = [0, 2, 3, 6, 7, 13, 16, 17, 21, 22]
    max_v = np.amax(velocity[:, main_nodes], axis=1) # (帧数,)
    max_v = max_v[:, np.newaxis] # (帧数,1)
    min_v = np.average(velocity[:, main_nodes], axis=1)  # (帧数,)
    min_v = min_v[:, np.newaxis]  # (帧数,1)
    velocity = np.concatenate([velocity, max_v, min_v], axis=1) # (帧数,24+2)

    #每帧加速度的最大值
    max_a = np.amax(acceleration[:, main_nodes], axis=1)  # (帧数,)
    max_a = max_a[:, np.newaxis]  # (帧数,1)
    min_a = np.average(acceleration[:, main_nodes], axis=1)  # (帧数,)
    min_a = min_a[:, np.newaxis]  # (帧数,1)
    acceleration = np.concatenate([acceleration, max_a, min_a], axis=1) # (帧数,24+2)

    # 每帧角速度的最大值
    main_nodes = [2, 3, 6, 7, 13, 16, 17, 21, 22]
    max_av = np.amax(angular_velocity[:, main_nodes], axis=1)  # (帧数,)
    max_av = max_av[:, np.newaxis]  # (帧数,1)
    min_av = np.average(angular_velocity[:, main_nodes], axis=1)  # (帧数,)
    min_av = min_av[:, np.newaxis]  # (帧数,1)
    angular_velocity = np.concatenate([angular_velocity, max_av, min_av], axis=1)  # (帧数,24+2)

    # 每帧角加速度的最大值
    max_aa = np.amax(angular_acceleration[:, main_nodes], axis=1)  # (帧数,)
    max_aa = max_aa[:, np.newaxis]  # (帧数,1)
    min_aa = np.average(angular_acceleration[:, main_nodes], axis=1)  # (帧数,)
    min_aa = min_aa[:, np.newaxis]  # (帧数,1)
    angular_acceleration = np.concatenate([angular_acceleration, max_aa, min_aa], axis=1)  # (帧数,24+2)

    """
    画图
    """
    # 关键几个部位，在全序列中的索引
    show_ids = [2, 3, 6, 7, 16, 17, 21, 22, 24, 25]
    show_names = ['l_knee', 'l_ankle', 'r_knee', 'r_ankle', 'l_elbow', 'l_wrist', 'r_elbow', 'r_wrist', 'frame_max', 'frame_avg']
    startidx = 8 # 0、2、4、6|8，左脚、右脚、左手、右手、最大最小
    num = 2
    is_show = [False, False, False, False]

    fig, a = plt.subplots(num, 1)
    x = np.linspace(0, frames - 1, frames)
    for i in range(num):
        # 速度
        if is_show[0]:
            v = velocity[:, show_ids[startidx + i]]
            a[i].plot(x, v, 'green', lw = 0.3)
            fv = filter(v)
            a[i].plot(x, fv*2, 'darkgreen', lw = 0.8)

        # 加速度
        if is_show[1]:
            v = acceleration[:, show_ids[startidx + i]] / 10.0
            a[i].plot(x, v, 'purple', lw = 0.3)
            fv = filter(v)
            a[i].plot(x, fv, 'pink', lw = 0.8)

        # 角速度
        if is_show[2]:
            v = angular_velocity[:, show_ids[startidx + i]]
            a[i].plot(x, v, 'orange', lw = 0.3)
            fv = filter(v)
            a[i].plot(x, fv, 'darkorange', lw = 0.8)

        # 角加速度
        if is_show[3]:
            v = angular_acceleration[:, show_ids[startidx + i]] / 10.0
            a[i].plot(x, v, 'yellow', lw=0.3)

        # strength
        #a[i].plot(x, strength, 'red', lw = 0.5)

        # peak
        #a[i].plot(x, peak_onehot, 'black', lw = 0.5)

        # beat
        a[i].plot(x, beat_onehot, 'black', lw = 0.5)

        #修正值
        if True:
            max_n_avg /= 10
            a[i].plot(x, max_n_avg, 'green', lw=0.3)
            fv0 = filter(max_n_avg)
            a[i].plot(x, fv0, 'darkgreen', lw=0.8)

            a[i].plot(x, momentum, 'blue', lw=0.3)
            fv1 = filter(momentum)
            a[i].plot(x, fv1, 'darkblue', lw=0.8)

            a[i].plot(x, energy, 'pink', lw=0.3)
            fv2 = filter(energy)
            a[i].plot(x, fv2, 'red', lw=0.8)


        a[i].set_title(show_names[startidx+i])
    plt.show()
