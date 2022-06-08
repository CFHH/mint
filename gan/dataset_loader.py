import tensorflow as tf

def load_dataset(batch_size, motion_input_length, motion_dim, data_files, num_cpu_threads=2):
    #files = tf.io.gfile.glob("../data/gan_with_trans/gan_train_tfrecord-*")

    name_to_features = {}
    name_to_features.update({
        "motion_name": tf.io.FixedLenFeature([], tf.string),
        "motion_sequence": tf.io.VarLenFeature(tf.float32),
        "motion_sequence_shape": tf.io.FixedLenFeature([2], tf.int64),
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
        return example

    def randomly_cut(example):
        motion_seq_length = tf.shape(example["motion_sequence"])[0] #帧数

        windows_size = motion_input_length
        # 若motion_seq_length = 100，windows_size = 20，则起始帧最大可以是80，因为[min,max)，所以max可以取81
        start = tf.random.uniform([], 0, motion_seq_length - windows_size + 1, dtype=tf.int32) #起始帧

        # motion input: [start, start + motion_input_length)
        example["motion_input"] = example["motion_sequence"][start:start + motion_input_length, :]
        example["motion_input"].set_shape([motion_input_length, motion_dim])

        del example["motion_sequence"]
        return example


    files = tf.io.gfile.glob(data_files)
    ds = tf.data.Dataset.from_tensor_slices(tf.constant(files))
    ds = ds.interleave(tf.data.TFRecordDataset, cycle_length=tf.data.AUTOTUNE, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False) #这里读取文件内容
    ds = ds.shuffle(100).repeat()
    ds = ds.map(decode_and_reshape_record, num_parallel_calls=num_cpu_threads)
    ds = ds.map(randomly_cut, num_parallel_calls=num_cpu_threads)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(1)
    return ds