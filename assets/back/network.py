import tensorflow as tf
import cv2
import numpy as np

# from keras_applications.vgg16 import VGG16
from com.mvc.model.modellocator import ModelLocator
from com.mvc.view.component.anchors import Anchor
from com.mvc.view.component.proposal import Proposal
from com.mvc.view.component.vgg16 import VGG16


class NetWork:
    vgg16 = None
    dataset = None

    _classes = ('__background__', 'jjy')
    _num_classes = len(_classes)

    def __init__(self, dataset):
        print("net work init ")
        self.dataset = dataset

        # 已经训练好的vgg16模型
        self.vgg16 = VGG16(ModelLocator.vgg16_model_path)

        for step, item in enumerate(dataset):
            self.__training(item)
            break
        pass

    def __analytical_data(self, data):
        # image.shape= [w,h,3]
        image = tf.cast(tf.image.decode_image(data["image"][0]), dtype=tf.float32) / 225.  # 归一化
        image = tf.expand_dims(image, 0)  # [w,h,3] -> [1,h,w,3]
        # print(image.shape)  # (1, 512, 583, 3)

        image_width = data["width"][0]
        image_height = data["height"][0]

        # 标签框 位置的百分比 需要乘上宽/高 才能具体出数字
        gt_x1 = tf.multiply(data["xmin"], tf.cast(image_width, tf.float32))
        gt_y1 = tf.multiply(data["ymin"], tf.cast(image_height, tf.float32))
        gt_x2 = tf.multiply(data["xmax"], tf.cast(image_width, tf.float32))
        gt_y2 = tf.multiply(data["ymax"], tf.cast(image_height, tf.float32))
        gt_label = tf.cast(data["label"], tf.float32)

        gt_boxes = tf.stack([gt_x1, gt_y1, gt_x2, gt_y2, gt_label], axis=1)
        return image, image_width, image_height, gt_boxes

    def __RPN(self, img, image_width, image_height):
        """
        Region Proposal Network
        :return:
        """
        # 使用已经训练好的vgg16网络提取图片特征 参考下载link:
        # https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5
        # 如果自动下载 保存的路径 "\.keras\models\"   (C:\Users\Alpx\.keras\models\vgg16_weights_tf_dim_ordering_tf_kernels.h5)
        # include_top (True): 是否包括模型的输出层。如果您根据自己的问题拟合模型，则不需要这些(设置成False)。
        # vgg16 = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=False)
        # img = tf.keras.applications.vgg16.preprocess_input(img)
        # print(img.shape)
        # # 下采样 16
        # feature = vgg16.predict(img)

        # 锚框
        boxes = Anchor().generate_anchors(image_width, image_height)

        # 获取特征图
        feature = self.vgg16.predict(img)
        proposal_model = Proposal()
        proposal_model.build(input_shape=feature.shape)
        res = proposal_model.predict(feature)

        print(res.shape)
        pass

    def __training(self, item):
        """
        训练网络
        :param item: 一组训练数据
        :return:
        """

        # 解析数据
        image, image_width, image_height, gt_boxes = self.__analytical_data(item)

        self.__RPN(image, image_width, image_height)
        pass
