import tensorflow as tf

def get_dataset_fn(batch_size, data_files, num_cpu_threads=2):
    #files = tf.io.gfile.glob("../data/tf_sstables/aist_generation_train_v2_tfrecord-*")

    modality_to_params = {}
    modality_to_params['motion'] = {}
    modality_to_params['motion']['feature_dim'] = 219 # 24个节点*(3x3的矩阵) + 3个平移
    modality_to_params['motion']['input_length'] = 120
    modality_to_params['motion']['target_length'] = 20
    modality_to_params['motion']['target_shift'] = 120
    modality_to_params['motion']['sample_rate'] = 1
    modality_to_params['motion']['resize'] = 0
    modality_to_params['motion']['crop_size'] = 0
    modality_to_params['audio'] = {}
    modality_to_params['audio']['feature_dim'] = 35
    modality_to_params['audio']['input_length'] = 240
    modality_to_params['audio']['target_length'] = 40
    modality_to_params['audio']['target_shift'] = 240
    modality_to_params['audio']['sample_rate'] = 2
    modality_to_params['audio']['resize'] = 0
    modality_to_params['audio']['crop_size'] = 0

    name_to_features = {}
    name_to_features.update({
        "motion_sequence": tf.io.VarLenFeature(tf.float32),
        "motion_sequence_shape": tf.io.FixedLenFeature([2], tf.int64),
        "motion_name": tf.io.FixedLenFeature([], tf.string),
        "audio_sequence": tf.io.VarLenFeature(tf.float32),
        "audio_sequence_shape": tf.io.FixedLenFeature([2], tf.int64),
        "audio_name": tf.io.FixedLenFeature([], tf.string),
    })


    # 转成tf.train.Example
    def decode_and_reshape_record(record):
        example = tf.io.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64: # tf.train.Example只支持tf.int64，但是TPU只支持tf.int32
                t = tf.dtypes.cast(t, tf.int32)
            example[name] = t
        example["motion_sequence"] = tf.reshape(tf.sparse.to_dense(example["motion_sequence"]), example["motion_sequence_shape"])
        example["audio_sequence"] = tf.reshape(tf.sparse.to_dense(example["audio_sequence"]), example["audio_sequence_shape"])
        return example


    def fact_preprocessing(example):
        motion_seq_length = tf.shape(example["motion_sequence"])[0] # 帧数x219

        motion_dim = modality_to_params["motion"]["feature_dim"]
        motion_input_length = modality_to_params["motion"]["input_length"]
        motion_target_length = modality_to_params["motion"]["target_length"]
        motion_target_shift = modality_to_params["motion"]["target_shift"]
        audio_dim = modality_to_params["audio"]["feature_dim"]
        audio_input_length = modality_to_params["audio"]["input_length"]

        pad_zeros = 1 #本来是6
        motion_dim += pad_zeros # 把平移数据从3个数字扩展成3x3的矩阵，没有意义
        example["motion_sequence"] = tf.pad(example["motion_sequence"], [[0, 0], [pad_zeros, 0]]) #上下左右，左边添pad_zeros个零

        windows_size = tf.maximum(motion_input_length, motion_target_shift + motion_target_length)
        #windows_size = tf.maximum(windows_size, audio_input_length)
        start = tf.random.uniform([], 0, motion_seq_length - windows_size + 1, dtype=tf.int32) #起始帧
        # 若motion_seq_length = 100，windows_size = 20，则起始帧最大可以是80，因为[min,max)，所以max可以取81

        # motion input: [start, start + motion_input_length)
        example["motion_input"] = example["motion_sequence"][start:start + motion_input_length, :]
        example["motion_input"].set_shape([motion_input_length, motion_dim])

        del example["motion_sequence"]
        del example["audio_sequence"]
        return example


    files = tf.io.gfile.glob(data_files)
    ds = tf.data.Dataset.from_tensor_slices(tf.constant(files))
    ds = ds.interleave(tf.data.TFRecordDataset, cycle_length=tf.data.AUTOTUNE, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False) #这里读取文件内容
    ds = ds.shuffle(100).repeat()
    ds = ds.map(decode_and_reshape_record, num_parallel_calls=num_cpu_threads)
    ds = ds.map(fact_preprocessing, num_parallel_calls=num_cpu_threads)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(1)
    return ds