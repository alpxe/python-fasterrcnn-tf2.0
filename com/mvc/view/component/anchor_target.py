import tensorflow as tf
import numpy as np
import numpy.random as npr
from com.so.bbox.cython_bbox import bbox_overlaps

from com.tool import util
from com.tool.bbox_transform import bbox_transform


def anchor_target_layer(rpn_cls_score, gt_boxes, image_width, image_height, anchors, K):
    total_anchors = anchors.shape[0]  # 锚框的个数
    print("锚框类别 K={0}".format(K))
    print("锚框个数 ={0}".format(total_anchors))
    print("锚点数量 ={0}".format(total_anchors / K))

    # map of shape (..., H, W)
    fh, fw = rpn_cls_score.shape[1:3]
    print("特征图 宽 x 高 = {0} x {1}\n".format(fw, fh))

    _allowed_border = 0

    # 不越界的锚框索引 一维数组
    inds_inside = np.where(
        (anchors[:, 0] >= -_allowed_border) &
        (anchors[:, 1] >= -_allowed_border) &
        (anchors[:, 2] < image_width + _allowed_border) &  # width
        (anchors[:, 3] < image_height + _allowed_border)  # height
    )[0]
    print("不越界的锚框个数={0}".format(len(inds_inside)))

    # 通过索引找到对应的锚框 筛选
    anchors = anchors[inds_inside, :]

    # 为锚框创建标签 1 is positive, 0 is negative, -1 is dont care
    labels = np.empty((len(inds_inside),), dtype=np.float32)
    labels.fill(-1)

    # labels的长度 与 anchor(P)的个数相关
    # 计算锚框与真实框的重合率 IoU
    # 二维表格： ，竖着是anchors 横向是gt_bboxes
    """
    IoU: overlaps(Anchor,GT)
              GT    GT    GT
    Anchor [value value value]
    Anchor [value value value]
    Anchor [value value value]
    """
    overlaps = bbox_overlaps(
        np.ascontiguousarray(anchors, np.float),
        np.ascontiguousarray(gt_boxes, np.float)
    )

    # 锚框 - 最优标定GT :  从标定GT中找到与该锚框IoU最大的值 找到每个锚框所对应的最优GT索引
    argmax_overlaps = overlaps.argmax(axis=1)  # 索引    len = anchors个数
    max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]  # IoU值

    # 标定GT - 最优的anchor : 从锚框中找到与该标定GT IoU最大的值
    gt_argmax_overlaps = overlaps.argmax(axis=0)  # exp: [6687] 即第6687个锚框与这个GT的IoU最大
    gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])]

    # 每个标定GT 肯定能找到最贴切的Anchor 用这个值去找更多的Anchor
    gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]
    """
    GT标框 与 所有锚框产生IoU 得到最大的值 可以标记为前景框
    IoU 与该值一样的，当然也可以作为前景框
    
    通过 锚框与GT标框 产生的IoU值 来确定 label的标签(-1:无关, 1:前景, 0:背景)
    """

    labels[max_overlaps < 0.3] = 0  # 设置背景
    labels[gt_argmax_overlaps] = 1  # 设置前景
    labels[max_overlaps >= 0.7] = 1

    # --平衡正负样本数量--
    total = 256  # 样本总数
    fg_index = np.where(labels == 1)[0]

    if len(fg_index) > total / 2:  # 正样本数过多
        disable_index = npr.choice(fg_index, size=int(len(fg_index) - total / 2), replace=False)
        labels[disable_index] = -1

    # 负样本的个数  经过上一步if判断后，labels正样本的个数 只会小于等于128
    num_bg = 256 - np.sum(labels == 1)
    print("正样本数目:{0}  负样本数:{1}".format((np.sum(labels == 1)), num_bg))

    # 从labels中获取 负样本的索引
    bg_index = np.where(labels == 0)[0]
    if len(bg_index) > num_bg:
        disable_index = npr.choice(bg_index, size=(len(bg_index) - num_bg), replace=False)
        labels[disable_index] = -1

    # --平衡正负样本数量-- 完成

    # 制作绑定框 bbox_targets
    bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)  # 初始化 内部权重
    bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)  # 初始化 外部权重

    # gt_boxes   Tensor -> numpy
    # 通过公式 当前的anchor与其最密切的绑定框之间的差距 计算出之间的差值 Delta Δ
    deltas = bbox_transform(anchors, gt_boxes.numpy()[argmax_overlaps, :])

    # labels是前景标记  bbox权重设置为(1.0, 1.0, 1.0, 1.0)
    bbox_inside_weights[labels == 1, :] = np.array((1.0, 1.0, 1.0, 1.0))

    # Give the positive RPN examples weight of p * 1 / {num positives}
    # and give negatives a weight of (1 - p)
    # Set to -1.0 to use uniform example weighting
    if True:  # 使用统一的示例加权
        num_examples = np.sum(labels >= 0)  # 256
        positive_weights = np.ones((1, 4)) * 1.0 / num_examples
        negative_weights = np.ones((1, 4)) * 1.0 / num_examples
    # else:
    #     RPN_POSITIVE_WEIGHT = 0.8  # RPN 正权重
    #     positive_weights = (RPN_POSITIVE_WEIGHT / np.sum(labels == 1))
    #     negative_weights = ((1.0 - RPN_POSITIVE_WEIGHT) / np.sum(labels == 0))
    #     pass
    bbox_outside_weights[labels == 1, :] = positive_weights
    bbox_outside_weights[labels == 0, :] = negative_weights

    # 子集数据转原始尺寸数据 len=fh*fw*K
    labels = util.unmap(labels, total_anchors, inds_inside, fill=-1)
    deltas = util.unmap(deltas, total_anchors, inds_inside, fill=0)
    bbox_inside_weights = util.unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
    bbox_outside_weights = util.unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)

    # --reshape--
    labels = labels.reshape((1, fh, fw, K)).transpose(0, 3, 1, 2)
    labels = labels.reshape((1, 1, K * fh, fw))  # 不懂为什么要reshape成这个结构?
    rpn_labels = labels

    deltas = deltas.reshape((1, fh, fw, K * 4))
    rpn_deltas = deltas

    bbox_inside_weights = bbox_inside_weights.reshape((1, fh, fw, K * 4))
    rpn_bbox_inside_weights = bbox_inside_weights

    bbox_outside_weights = bbox_outside_weights.reshape((1, fh, fw, K * 4))
    rpn_bbox_outside_weights = bbox_outside_weights

    return rpn_labels, rpn_deltas, rpn_bbox_inside_weights, rpn_bbox_outside_weights
