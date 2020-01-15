from com.mvc.app import App
from com.tool.anchor import Anchor
from com.mvc.model.net import faster_rcnn
import numpy as np


class NeuralNetworks:
    def __init__(self, dataset):
        print("NeuralNetworks")
        # self.vgg16 = VGG16(ModelLocator.vgg16_model_path)

        self.model = faster_rcnn.FasterRCNN()

        for step, item in enumerate(dataset):
            self.__training(item)
            break

    def __training(self, data):
        image, image_width, image_height, gt_boxes = App().dataset_proxy.analytical(data)

        x = self.model(image_width, image_height)

        print(x)
        # self.stride = 16  # 下采样
        # self.anchor_scales = 2 ** np.arange(3, 6)  # [8 16 32]
        # self.anchor_ratios = [0.5, 1, 2]
        #
        # Anchor(self.anchor_scales, self.anchor_ratios, self.stride)
        # anchors = Anchor().generate_anchors(image_width, image_height)
        # print(anchors)
        pass
