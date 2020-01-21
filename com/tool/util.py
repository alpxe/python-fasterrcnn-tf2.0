import tensorflow as tf
import numpy as np


def rpn_cls_softmax(data, dims):
    input_shape = tf.shape(data)  # [-1,h,w,18]           (1,  32, 36,18)

    data = reshape_layer(data, dims)  # [-1,9*h,w,2]    (1, 288, 36, 2)
    data = __softmax_layer(data)
    data = reshape_layer(data, input_shape[-1])  # (1,32,36,18)
    # print(data[:, :, :, 0] + data[:, :, :, 9])
    return data


def reshape_layer(data, num_dim):
    """
    [-1,h,w,18] -> [-1,h,w,2*9] -> [-1,h*9,w,2]
    :param data:
    :param num_dim:
    :return:
    """
    input_shape = tf.shape(data)
    # change the channel to the caffe format
    to_caffe = tf.transpose(data, [0, 3, 1, 2])
    # then force it to have channel 2
    reshaped = tf.reshape(to_caffe, tf.concat(axis=0, values=[[1, num_dim, -1], [input_shape[2]]]))
    # then swap the channel back
    to_tf = tf.transpose(reshaped, [0, 2, 3, 1])
    return to_tf


def __softmax_layer(data):
    input_shape = tf.shape(data)
    reshaped = tf.reshape(data, [-1, input_shape[-1]])  # (288x36,2) -> (10368, 2)
    score = tf.nn.softmax(reshaped)
    return tf.reshape(score, input_shape)


def unmap(data, count, inds, fill=0):
    """
    将项目的子集数据 隐射到原始大小数据集
    :param data: 子集数据
    :param count: 原始集总数
    :param inds: 子集数据所在索引
    :param fill: 填充的数值
    :return:
    """
    if len(data.shape) == 1:  # 如果是一维数组
        ret = np.empty((count,), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret
