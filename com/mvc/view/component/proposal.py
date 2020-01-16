import tensorflow as tf

from com.tool.bbox_transform import bbox_transform_inv, clip_boxes


def proposal_layer(rpn_cls_prob, rpn_bbox_pred, image_width, image_height, anchors, K):
    """
    提取框
    :return:
    """
    scores = rpn_cls_prob[:, :, :, K:]  # 前景打分概率值
    scores = tf.reshape(scores, shape=(-1,))

    rpn_bbox_pred = tf.reshape(rpn_bbox_pred, shape=(-1, 4))

    # 预测框     N个锚框 x N个预测detals = N个预测框
    proposals = bbox_transform_inv(anchors, rpn_bbox_pred)
    proposals = clip_boxes(proposals, image_width, image_height)  # 修正超出边界的框

    # Non-maximal suppression 非极大值抑制  返回的是一个索引的集合
    indices = tf.image.non_max_suppression(proposals, scores, max_output_size=2000, iou_threshold=0.7)

    boxes = tf.gather(proposals, indices)
    scores = tf.gather(scores, indices)
    scores = tf.reshape(scores, shape=(-1, 1))

    batch_inds = tf.zeros((tf.shape(indices)[0], 1), dtype=tf.float32)
    blob = tf.concat([batch_inds, boxes], 1)  # 一组[ 0 ,x1,y1,x2,y2]

    return blob, scores
