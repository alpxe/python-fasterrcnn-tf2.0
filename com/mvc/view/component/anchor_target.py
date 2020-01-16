import tensorflow as tf
import numpy as np

from com.so.bbox.cython_bbox import bbox_overlaps


def anchor_target_layer(rpn_cls_score, gt_boxes, image_width, image_height, anchors, K):
    print("锚框类别={0}".format(K))
    print("锚框个数={0}".format(anchors.shape[0]))
    print("锚点数量={0}".format(anchors.shape[0] / K))

    # map of shape (..., H, W)
    fh, fw = rpn_cls_score.shape[1:3]
    print("特征图 宽 x 高 = {0} x {1}".format(fw, fh))

    _allowed_border = 0

    # 不越界的锚框索引 一维数组
    inds_inside = np.where(
        (anchors[:, 0] >= -_allowed_border) &
        (anchors[:, 1] >= -_allowed_border) &
        (anchors[:, 2] < image_width + _allowed_border) &  # width
        (anchors[:, 3] < image_height + _allowed_border)  # height
    )[0]

    # 通过索引找到对应的锚框
    anchors = anchors[inds_inside, :]

    # 为锚框创建标签 1 is positive, 0 is negative, -1 is dont care
    labels = np.empty((len(inds_inside),), dtype=np.float32)
    labels.fill(-1)

    # labels的长度 与 anchor(P)的个数相关
    # 计算锚框与真实框的重合率 IoU
    # 二维表格： ，竖着是anchors 横向是gt_bboxes
    """
    [ P [Q Q Q Q],
      P [Q Q Q Q],
      P [Q Q Q Q] ]   overlaps[P, Q]
    """
    overlaps = bbox_overlaps(
        np.ascontiguousarray(anchors, np.float),
        np.ascontiguousarray(gt_boxes, np.float)
    )

    # 锚框 - 最优标定GT :  从标定GT中找到与该锚框IoU最大的值
    argmax_overlaps = overlaps.argmax(axis=1)  # 索引    len = anchors个数
    max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]  # IoU值

    # 标定GT - 最优的anchor : 从锚框中找到与该标定GT IoU最大的值
    gt_argmax_overlaps = overlaps.argmax(axis=0)
    gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])]

    gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

    np.set_printoptions(threshold=np.inf, precision=5, suppress=True)
    print(overlaps)
    print(gt_max_overlaps)
    print(gt_argmax_overlaps)

    pass
