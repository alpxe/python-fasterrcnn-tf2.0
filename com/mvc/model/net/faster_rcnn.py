import tensorflow as tf
import numpy as np
from com.tool import anchor
from com.mvc.model.net import rpn


class FasterRCNN(tf.keras.Model):
    def __init__(self):
        super(FasterRCNN, self).__init__()

        self.stride = 16  # 下采样
        self.anchor_scales = 2 ** np.arange(3, 6)  # [8 16 32]
        self.anchor_ratios = [0.5, 1, 2]

        self.K = len(self.anchor_scales) * len(self.anchor_ratios)  # 3x3=9

        self.rpn_net = rpn.RPN()

    def call(self, image_width, image_height):
        anchors = anchor.Anchor(
            self.anchor_scales, self.anchor_ratios, self.stride
        ).generate_anchors(image_width, image_height)

        return anchors
