def set_session_config(per_process_gpu_memory_fraction=None, allow_growth=None):
    """tensorflowのオプションの設定"""
    import tensorflow as tf
    import keras.backend as K

    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            per_process_gpu_memory_fraction=per_process_gpu_memory_fraction,
            allow_growth=allow_growth,
        )
    )
    sess = tf.Session(config=config)
    K.set_session(sess)
