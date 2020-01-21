import tensorflow as tf
import numpy as np

from com.mvc.model.modellocator import ModelLocator
from com.mvc.model.net import rpn


class FasterRCNN(tf.keras.Model):
    def __init__(self, num_classes):
        super(FasterRCNN, self).__init__()

        self.stride = ModelLocator.stride
        self.anchor_scales = ModelLocator.anchor_scales
        self.anchor_ratios = ModelLocator.anchor_ratios

        self.rpn_net = rpn.RPN(
            scales=self.anchor_scales,
            ratios=self.anchor_ratios,
            stride=self.stride,
            num_classes=num_classes
        )

    def call(self, img, image_width, image_height, gt_boxes):
        return self.rpn_net(img, image_width, image_height, gt_boxes)

        # anchors = anchor.Anchor(
        #     self.anchor_scales, self.anchor_ratios, self.stride
        # ).generate_anchors(image_width, image_height)

        # return anchors
