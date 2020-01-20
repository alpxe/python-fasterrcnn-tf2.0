from com.tool.bbox_transform import bbox_transform_inv, clip_boxes, bbox_transform
from com.so.bbox.cython_bbox import bbox_overlaps
import tensorflow as tf
import numpy as np
import numpy.random as npr


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


def proposal_target_layer(rois, rpn_scores, gt_boxes, num_classes):
    # ROIs shape = (0,x1,y1,x2,y2)
    all_rois = rois  # 2000个
    all_scores = rpn_scores

    # 候选集中包括真实框
    if True:
        # 勾选框中加入真实框
        zeros = np.zeros((gt_boxes.shape[0], 1))
        all_rois = np.vstack(
            (all_rois, np.hstack((zeros, gt_boxes[:, :-1])))
        )
        all_scores = np.vstack((all_scores, zeros))

    num_images = 1
    BATCH_SIZE = 128  # Minibatch size (number of regions of interest [ROIs])
    FG_FRACTION = 0.25  # Fraction of minibatch that is labeled foreground (i.e. class > 0)
    rois_per_image = BATCH_SIZE / num_images  # 128
    fg_rois_per_image = np.round(FG_FRACTION * rois_per_image)  # 32

    __sample_rois(
        all_rois=all_rois,
        all_scores=all_scores,
        gt_boxes=gt_boxes,
        fg_rois_per_image=fg_rois_per_image,
        rois_per_image=rois_per_image,
        num_classes=num_classes
    )
    pass


def __sample_rois(all_rois, all_scores, gt_boxes, fg_rois_per_image, rois_per_image, num_classes):
    """
    生成包含前景和背景的RoI随机样本例子
    """
    overlaps = bbox_overlaps(
        np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
        np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float)
    )

    # 2000提取框 - 最优标定GT :  从标定GT中找到与该提取框IoU最大的值
    gt_assignment = overlaps.argmax(axis=1)  # 索引
    max_overlaps = overlaps.max(axis=1)  # 值

    labels = gt_boxes.numpy()[gt_assignment, 4]  # 真实标签 ==>2000个1
    print("IoU 重合率 大于 0.5的 个数 = {0}".format(np.where(max_overlaps > 0.5)[0].size))

    FG_THRESH = 0.5  # 前景阈值
    BG_THRESH_HI = 0.5  # 背景 区间
    BG_THRESH_LO = 0.1  # 背景 区间
    fg_index = np.where(max_overlaps >= FG_THRESH)[0]  # 提取框与与GT标框值大于0.5 视为前景框
    bg_index = np.where((max_overlaps < BG_THRESH_HI) & (max_overlaps >= BG_THRESH_LO))[0]  # 0.1< x <0.5 为背景框

    # Small modification to the original version where we ensure a fixed number of regions are sampled
    if fg_index.size > 0 and bg_index.size > 0:
        fg_rois_per_image = min(fg_rois_per_image, fg_index.size)
        fg_index = npr.choice(fg_index, size=int(fg_rois_per_image), replace=False)
        bg_rois_per_image = rois_per_image - fg_rois_per_image
        to_replace = bg_index.size < bg_rois_per_image
        bg_index = npr.choice(bg_index, size=int(bg_rois_per_image), replace=to_replace)
    elif fg_index.size > 0:
        to_replace = fg_index.size < rois_per_image  # 如果前景框的数量不超过 128个 靠重复随机去填满128个
        fg_index = npr.choice(fg_index, size=int(rois_per_image), replace=to_replace)
        fg_rois_per_image = rois_per_image  # 前景样本数等于总样本数
    elif bg_index.size > 0:
        to_replace = bg_index.size < rois_per_image
        bg_index = npr.choice(bg_index, size=int(rois_per_image), replace=to_replace)
        fg_rois_per_image = 0
    else:
        print("无--提取框与与GT标框值")

    print("proposal:")
    print("正样本数目:{0}  负样本数:{1}".format(fg_index.size, bg_index.size))
    keep_index = np.append(fg_index, bg_index)  # 索引
    labels = labels[keep_index]
    labels[int(fg_rois_per_image):] = 0  # 因为后面拼接的是背景 所以可以明确的将值设置为0

    rois = all_rois[keep_index]  # 最优的前后背景 提取框
    roi_scores = all_scores[keep_index]

    # [标签,Δx,Δy,Δw,Δh]
    bbox_deltas_data = _compute_targets(rois[:, 1:5], gt_boxes.numpy()[gt_assignment[keep_index], :4], labels)
    bbox_deltas, bbox_inside_weights = _get_bbox_regression_labels(bbox_deltas_data, num_classes)

    """
        Returns:
         labels:  [1,2,3,...0,0,0,0] IoU最优的框 阈值大于设定值(0.5) 绑定框的labels
         rois:  前后背景 提取框
         rois_scores:  对应的分数
         bbox_targets:   前景框数个 x [0,0,0,0  ,0,0,0,0  ,0,0,0,0  ,...] 4个一组 num_class几个就几组 对应的类别存在对应类别的位置
                         值为 Delta Δ
         bbox_inside_weights:  权重与bbox_targets 类似. 但是值为 ..., 1,1,1,1,  ...代表权重为1
        """
    return labels, rois, roi_scores, bbox_deltas, bbox_inside_weights


def _compute_targets(ex_rois, gt_rois, labels):
    """
    计算图像边界回归框的目标
    :param ex_rois: 预测框
    :param gt_rois: 真实框
    :param labels: 真实标签
    :return:
    """
    assert ex_rois.shape[0] == gt_rois.shape[0]  # 个数一致
    assert ex_rois.shape[1] == 4  # 4个坐标数值 x1,y1,x2,y2
    assert gt_rois.shape[1] == 4  # 4个坐标数值

    # 通过公式 当前的预测框 与其最密切的绑定框 之间的增量 计算出之间的差值 Delta Δ
    targets = bbox_transform(ex_rois, gt_rois)  # Delta Δ

    BBOX_NORMALIZE_MEANS = (0.0, 0.0, 0.0, 0.0)
    BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)
    targets = ((targets - np.array(BBOX_NORMALIZE_MEANS)) / np.array(BBOX_NORMALIZE_STDS))

    # [label, dx, dy, dw, dh] 这里的label是 绑定框的类别值 0代表背景，1，2，3，4....表示对应某种东西的类别
    return np.hstack((labels[:, np.newaxis], targets)).astype(np.float32, copy=False)


def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets (bbox_target_data) are stored in a
    compact form N x (class, tx, ty, tw, th)

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets).

    Returns:
        bbox_target (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """

    clss = bbox_target_data[:, 0]  # 取出类别
    # 前景框个数 x  numclass个0000  即每组框坐标(0000) 对应1个类别 由下面for ind in inds 体现规划
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)

    inds = np.where(clss > 0)[0]  # 代表前景的索引

    # 对应的类别分配到对应位置的坐标组
    for ind in inds:
        cls = clss[ind]
        start = int(4 * cls)
        end = start + 4

        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = (1.0, 1.0, 1.0, 1.0)  # 因为是前景值，所以权重是1
        pass

    return bbox_targets, bbox_inside_weights
