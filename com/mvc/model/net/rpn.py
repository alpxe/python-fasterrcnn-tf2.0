import tensorflow as tf

from com.mvc.view.component.anchor_target import anchor_target_layer
from com.mvc.view.component.proposal import proposal_layer
from com.tool import anchor, util


class RPN(tf.keras.Model):
    def __init__(self, scales, ratios, stride):
        super(RPN, self).__init__()

        self.anchor_scales = scales  # [8 16 32]
        self.anchor_ratios = ratios  # [0.5, 1, 2]
        self.stride = stride  # 16

        self.K = len(scales) * len(ratios)  # 3x3=9

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
        anchors = anchor.Anchor(
            self.anchor_scales, self.anchor_ratios, self.stride
        ).generate_anchors(image_width, image_height)

        shared = self.shared_layer(inputs)
        rpn_cls_score = self.rpn_cls_score(shared)  # RPN 分类 预测值
        rpn_bbox_pred = self.rpn_bbox_pred(shared)  # RPN BBOX 预测值

        # RPN 分类 预测值 -> 概率 (1,32,36,18) 0-8是正标签 9-18是负标签
        rpn_cls_prob = util.rpn_cls_softmax(rpn_cls_score, 2)  # RPN 分类 概率

        # ROIs 指的是Selective Search的输出   rois=[[0,x1,y1,x2,y2]...] rpn_scores=[[前景得分(大于0.7)]]
        rois, rpn_scores = proposal_layer(rpn_cls_prob, rpn_bbox_pred, image_width, image_height, anchors, self.K)

        anchor_target_layer(rpn_cls_score, gt_boxes, image_width, image_height, anchors, self.K)
        pass
