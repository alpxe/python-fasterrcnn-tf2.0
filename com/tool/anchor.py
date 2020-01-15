from com.core.base.singleton import Singleton
import tensorflow as tf
import numpy as np


class Anchor(Singleton):
    def __single__(self, scales, ratios, stride):
        self.scales = scales
        self.ratios = ratios
        self.stride = stride

    def generate_anchors(self, image_width, image_height):
        """
        生成锚框
        :param image_width: 图片宽度
        :param image_height: 图片高度
        :return: (w/16) x (h/16) x 9 个锚框
        """

        # 特征图宽高
        feature_width = tf.cast(tf.math.ceil(image_width / self.stride), dtype=tf.int32)
        feature_height = tf.cast(tf.math.ceil(image_height / self.stride), dtype=tf.int32)

        # 生成网格点坐标矩阵  x=[8 16 32] y=[0.5,1,2] 3x3=9个锚框
        scales, ratios = np.meshgrid(self.scales, self.ratios)
        scales, ratios = scales.flatten(), ratios.flatten()  # 平铺

        side_width = scales * np.sqrt(ratios)  # 9个宽边
        side_height = scales / np.sqrt(ratios)  # 9个高边  一一对应

        shifts_x = np.arange(0, feature_width) * self.stride  # 中心x
        shifts_y = np.arange(0, feature_height) * self.stride  # 中心y
        print("锚框个数{0}".format(feature_width * feature_height * 9))

        # 组合中心坐标
        shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)
        shifts_x, shifts_y = shifts_x.flatten(), shifts_y.flatten()

        center_x, anchor_w = np.meshgrid(shifts_x, side_width)
        center_y, anchor_h = np.meshgrid(shifts_y, side_height)
        anchor_center = np.stack([center_x, center_y], axis=2).reshape(-1, 2)
        anchor_size = np.stack([anchor_w, anchor_h], axis=2).reshape(-1, 2)
        # 左上 右下 坐标输出
        boxes = np.concatenate([anchor_center - 0.5 * anchor_size, anchor_center + 0.5 * anchor_size], axis=1)

        return boxes
#
#     pass
#
#
# if __name__ == "__main__":
#     import numpy as np
#
#     stride = 16  # 下采样
#     anchor_scales = 2 ** np.arange(3, 6)  # [8 16 32]
#     anchor_ratios = [0.5, 1, 2]
#
#     anchor = Anchor(anchor_scales, anchor_ratios, stride)
#     anchor.generate_anchors(tf.constant(550), tf.constant(400))
