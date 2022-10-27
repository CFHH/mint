from absl import app
from absl import flags
import glob
import tqdm
from moviepy.editor import *

FLAGS = flags.FLAGS
flags.DEFINE_string('mp4_path', 'D:/mint-video/', 'mp4 path')
flags.DEFINE_string('mp3_save_path', 'D:/mint-mp3/', 'mp3 path')

def process(mp4file):
    video_name = os.path.basename(mp4file)
    video_name = video_name.split('.')[0]
    video_name = video_name.replace('_c01_', '_cAll_')
    mp3file = os.path.join(FLAGS.mp3_save_path, "%s.mp3" % video_name)

    video = VideoFileClip(mp4file)
    audio = video.audio
    audio.write_audiofile(mp3file)
    return

def main(_):
    mp4_files = glob.glob(os.path.join(FLAGS.mp4_path, "*_*_c01_*_*_*.mp4"))
    for file in tqdm.tqdm(mp4_files):
        print("Process %s" % file)
        process(file)
    return

if __name__ == '__main__':
    app.run(main)


