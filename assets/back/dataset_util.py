import tensorflow as tf

"""

"""


def int64_feature(value):
    if type(value) is int:
        value = [value]

    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float64_feature(value):
    if type(value) is float:
        value = [value]

    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    if type(value) is bytes:
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def string_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode('utf-8')]))
