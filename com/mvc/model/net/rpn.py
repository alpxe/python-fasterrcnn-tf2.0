import tensorflow as tf

from com.mvc.view.component.anchor_target import anchor_target_layer
from com.mvc.view.component.proposal import proposal_layer, proposal_target_layer
from com.tool import anchor, util


class RPN(tf.keras.Model):
    def __init__(self, scales, ratios, stride, num_classes):
        super(RPN, self).__init__()

        self.anchor_scales = scales  # [8 16 32]
        self.anchor_ratios = ratios  # [0.5, 1, 2]
        self.stride = stride  # 16
        self.num_classes = num_classes  # 总类别

        self.K = len(scales) * len(ratios)  # 3x3=9

        # 初始化 Anchor 参数
        anchor.Anchor(self.anchor_scales, self.anchor_ratios, self.stride)

        # 共享层  He正态分布初始化
        self.shared_layer = tf.keras.layers.Conv2D(
            256, (3, 3), padding='same', kernel_initializer='he_normal', name='shared_layer'
        )

        self.rpn_cls_score = tf.keras.layers.Conv2D(
            2 * self.K, (1, 1), padding='valid', kernel_initializer='he_normal', name='rpn_cls_score'
        )

        self.rpn_bbox_pred = tf.keras.layers.Conv2D(
            4 * self.K, (1, 1), padding='valid', kernel_initializer='he_normal', name='rpn_bbox_pred'
        )

    def call(self, inputs, image_width, image_height, gt_boxes):
        # 创建锚框
        anchors = anchor.Anchor().generate_anchors(image_width, image_height)

        shared = self.shared_layer(inputs)  # 共享层
        rpn_cls_score = self.rpn_cls_score(shared)  # RPN 分类 预测值 (1, 32, 36, 18)
        rpn_bbox_pred = self.rpn_bbox_pred(shared)  # RPN BBOX 预测值 (1, 32, 36, 36)

        # RPN 分类 预测值 -> 概率 (1,32,36,18) 0-8是正标签 9-18是负标签
        rpn_cls_prob = util.rpn_cls_softmax(rpn_cls_score, 2)  # RPN 分类 概率

        # ROIs 指的是Selective Search的输出   rois=[[0,x1,y1,x2,y2]...] rpn_scores=[[前景得分(值一般大于0.7)]]
        rois, rpn_scores = proposal_layer(rpn_cls_prob, rpn_bbox_pred, image_width, image_height, anchors, self.K)
        """
         2K卷积是标签值(前景+背景) 4K卷积是坐标值Δ  通过非极大值抑制((坐标值Δ+锚框),(标签纸),2000,0.7)
          rois = 2000个 of (坐标值Δ+锚框=提取框)
          rpn_scores = 2000个 of 前景概率值
        """

        rpn_labels, rpn_deltas, rpn_bbox_inside_weights, rpn_bbox_outside_weights = anchor_target_layer(
            rpn_cls_score, gt_boxes, image_width, image_height, anchors, self.K)
        """
        rpn_labels = fh x fw x K 个 include 256 个标签(前景+背景) 其他为-1
        rpn_deltas = 锚框detal
        rpn_bbox_inside_weights =  内部权重(1.0, 1.0, 1.0, 1.0) where(labels == 1)
        rpn_bbox_outside_weights = 外部权重(1.0 / num_examples) where(labels >= 0)
        """

        # [-1,h,w,18] -> [-1,h,w,2*9] -> [-1,h*9,w,2]
        rpn_cls_score_reshape = tf.reshape(util.reshape_layer(rpn_cls_score, 2), [-1, 2])  # (1,288,36,2)->(10368, 2)
        rpn_label = tf.reshape(rpn_labels, [-1])  # (1, 1, 288, 36)-> 一维10368
        rpn_select = tf.where(tf.not_equal(rpn_label, -1))  # 比较运算符 rpn_label!=-1?True:False 找到前景与背景索引
        rpn_cls = tf.reshape(tf.gather(rpn_cls_score_reshape, rpn_select), [-1, 2])  # 特征值
        rpn_label = tf.reshape(tf.gather(rpn_label, rpn_select), [-1])  # 1 或 0

        rpn_cross_entropy = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls, labels=tf.cast(rpn_label, dtype=tf.int64))
        )

        return rpn_cross_entropy

        # labels, rois, roi_scores, bbox_deltas, bbox_inside_weights = proposal_target_layer(
        #     rois, rpn_scores, gt_boxes, self.num_classes)
        # # labels, rois, roi_scores, bbox_deltas, bbox_inside_weights = proposal_target_layer(
        # #     rois, rpn_scores, gt_boxes, self.num_classes)
        # """
        # 2000 ->128
        # labels 具体标签
        # rois (坐标值Δ+锚框)
        # roi_scores (前景值)
        # bbox_deltas 4*num_classes宽度的 前景deltas值
        # bbox_inside_weights 4*num_classes宽度的 权重值 1
        # """
        #
        # return rois  # (坐标值Δ+锚框)
