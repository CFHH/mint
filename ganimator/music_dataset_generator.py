from absl import app
from absl import flags
from absl import logging
import os
import numpy as np
import math
import tqdm
import glob
import json
import librosa

FLAGS = flags.FLAGS
flags.DEFINE_string('music_dir', 'D:/mint-music/', 'Path to the AIST wav files.')
flags.DEFINE_string('video_dir', 'D:/mint-video/', 'Path to the AIST mp4 files.')

def get_bpm(audio_name):
    """Get tempo (BPM) for a music by parsing music name."""
    assert len(audio_name) == 4
    if audio_name[0:3] in ['mBR', 'mPO', 'mLO', 'mMH', 'mLH', 'mWA', 'mKR', 'mJS', 'mJB']:
        return int(audio_name[3]) * 10 + 80
    elif audio_name[0:3] == 'mHO':
        return int(audio_name[3]) * 5 + 110
    else:
        assert False, audio_name

def get_music_feature(file_name):
    FPS = 60
    HOP_LENGTH = 512
    SR = FPS * HOP_LENGTH
    EPS = 1e-6

    motion_name = os.path.basename(file_name)
    motion_name = motion_name.split('.')[0]
    motion_name = motion_name.replace('_c01_', '_cAll_')
    audio_name = motion_name.split("_")[4]

    # _ = SR = 60 * 512，len(data) / _ = 秒数
    data, _ = librosa.load(file_name, sr=SR)

    # envelope.shape = (帧数,)，帧数 = FPS * 秒数 + 1
    envelope = librosa.onset.onset_strength(data, sr=SR)

    # mfcc.shape = (帧数, 20)
    mfcc = librosa.feature.mfcc(data, sr=SR, n_mfcc=20).T

    # chroma.shape = (帧数, 12)
    chroma = librosa.feature.chroma_cens(data, sr=SR, hop_length=HOP_LENGTH, n_chroma=12).T

    # peak_onehot.shape = (帧数,)
    peak_idxs = librosa.onset.onset_detect(onset_envelope=envelope.flatten(), sr=SR, hop_length=HOP_LENGTH)
    peak_onehot = np.zeros_like(envelope, dtype=np.float32)
    peak_onehot[peak_idxs] = 1.0

    # beat_onehot.shape = (帧数,)
    tempo, beat_idxs = librosa.beat.beat_track(
        onset_envelope=envelope, sr=SR, hop_length=HOP_LENGTH,
        start_bpm=get_bpm(audio_name), tightness=100)
    beat_onehot = np.zeros_like(envelope, dtype=np.float32)
    beat_onehot[beat_idxs] = 1.0

    # beat_onehot.shape = (帧数,)
    """
    audio_feature = np.concatenate([
        envelope[:, None], mfcc, chroma, peak_onehot[:, None], beat_onehot[:, None]
    ], axis=-1)
    """
    #audio_feature = np.concatenate([envelope[:, None], peak_onehot[:, None], beat_onehot[:, None]], axis=-1)
    audio_feature = {
        'name': motion_name,
        'frames': envelope.shape[0],
        'strength': envelope.tolist(),
        'peak_idx': peak_idxs.tolist(),
        'beat_idx': beat_idxs.tolist()
    }

    return audio_feature


def main(_):
    json_data = {}

    music_files = glob.glob(os.path.join(FLAGS.video_dir, "*_*_c01_*_*_*.mp4"))
    for file_name in tqdm.tqdm(music_files):
        print("Process %s" % file_name)
        audio_feature = get_music_feature(file_name)
        json_data[audio_feature['name']] = audio_feature

    json_str = json.dumps(json_data)
    json_file = os.path.join(FLAGS.video_dir, "music_feature.dat")
    with open(json_file, 'w') as f:
        f.write(json_str)


if __name__ == '__main__':
    app.run(main)