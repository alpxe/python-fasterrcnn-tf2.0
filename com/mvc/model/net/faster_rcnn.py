import tensorflow as tf
import numpy as np
from com.tool import anchor
from com.mvc.model.net import rpn


class FasterRCNN(tf.keras.Model):
    def __init__(self, num_classes):
        super(FasterRCNN, self).__init__()

        self.stride = 16  # 下采样
        self.anchor_scales = 2 ** np.arange(5, 8)  # [8 16 32]
        self.anchor_ratios = [0.5, 1, 2]

        self.rpn_net = rpn.RPN(
            scales=self.anchor_scales,
            ratios=self.anchor_ratios,
            stride=self.stride,
            num_classes=num_classes
        )

    def call(self, img, image_width, image_height, gt_boxes):
        self.rpn_net(img, image_width, image_height, gt_boxes)
        pass
        # anchors = anchor.Anchor(
        #     self.anchor_scales, self.anchor_ratios, self.stride
        # ).generate_anchors(image_width, image_height)

        # return anchors
