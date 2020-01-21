import tensorflow as tf
import numpy as np


def bbox_transform(ex_rois, gt_rois):
    ex_ctr_x, ex_ctr_y, ex_widths, ex_heights = __port2rect(ex_rois)
    gt_ctr_x, gt_ctr_y, gt_widths, gt_heights = __port2rect(gt_rois)
    print(gt_widths)

    dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    dw = np.log(gt_widths / ex_widths)
    dh = np.log(gt_heights / ex_heights)

    # [[x...],      [[x,y,w,h],
    #  [y...],       [x,y,w,h],
    #  [w...],  ==>  [x,y,w,h],
    #  [h...]]       ......   ]
    deltas = np.vstack((dx, dy, dw, dh)).transpose()
    return deltas


def bbox_transform_inv(boxes, deltas):
    """
    矩形 + Delta = 新的矩形
    boxes 与 deltas 长度一致
    :param boxes: 矩形框 左上,右下(x1,y1,x2,y2)
    :param deltas: Delta Δ，表示增量
    :return: 矩形框 左上,右下(x1,y1,x2,y2)
    """

    boxes = tf.cast(boxes, deltas.dtype)

    ctr_x, ctr_y, widths, heights = __port2rect(boxes)

    dx = deltas[:, 0]
    dy = deltas[:, 1]
    dw = deltas[:, 2]
    dh = deltas[:, 3]

    pred_ctr_x = tf.add(tf.multiply(dx, widths), ctr_x)
    pred_ctr_y = tf.add(tf.multiply(dy, heights), ctr_y)
    pred_w = tf.multiply(tf.exp(dw), widths)
    pred_h = tf.multiply(tf.exp(dh), heights)

    pred_boxes0 = tf.subtract(pred_ctr_x, pred_w * 0.5)
    pred_boxes1 = tf.subtract(pred_ctr_y, pred_h * 0.5)
    pred_boxes2 = tf.add(pred_ctr_x, pred_w * 0.5)
    pred_boxes3 = tf.add(pred_ctr_y, pred_h * 0.5)

    return tf.stack([pred_boxes0, pred_boxes1, pred_boxes2, pred_boxes3], axis=1)


def __port2rect(boxes):
    """
    左上右下 转 中心点+宽高
    """
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0

    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    return ctr_x, ctr_y, widths, heights


def clip_boxes(boxes, w, h):
    """
    将超出范围的框 超出的部分裁剪
    :param boxes:
    :param w: 原始图像的宽
    :param h: 原始图像的高
    :return:
    """
    w = tf.cast(w, dtype=tf.float32)
    h = tf.cast(h, dtype=tf.float32)

    b0 = tf.maximum(tf.minimum(boxes[:, 0], w - 1), 0)
    b1 = tf.maximum(tf.minimum(boxes[:, 1], h - 1), 0)
    b2 = tf.maximum(tf.minimum(boxes[:, 2], w - 1), 0)
    b3 = tf.maximum(tf.minimum(boxes[:, 3], h - 1), 0)
    return tf.stack([b0, b1, b2, b3], axis=1)
