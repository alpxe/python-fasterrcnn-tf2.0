from com.mvc.app import App
from com.mvc.model.modellocator import ModelLocator
from com.mvc.model.net.vgg16 import VGG16
from com.mvc.model.net import faster_rcnn
import tensorflow as tf


class NeuralNetworks:
    def __init__(self, dataset):
        print("NeuralNetworks\n")

        self.num_classes = len(ModelLocator.CLASSES)  # 识别种类

        self.vgg16 = VGG16(ModelLocator.vgg16_model_path)

        self.model = faster_rcnn.FasterRCNN(self.num_classes)

        # 优化器
        self.optimizer = tf.optimizers.Adam(1e-4)

        for step, item in enumerate(dataset):
            loss = self.__training(item)
            # if step == 1:
            # break
            print(loss)

    def __training(self, data):
        # 解析一组数据  图片源  图片宽度 图片高度 GT标框
        image, image_width, image_height, gt_boxes = App().dataset_proxy.analytical(data)

        # 提取图片特征 下采样16
        img = self.vgg16.predict(image)  # (1,32,36,512)

        with tf.GradientTape() as tape:
            loss = self.model(img, image_width, image_height, gt_boxes)

        # 更新训练的变量 trainable_variables
        trainable_variables = self.model.trainable_variables

        # Compute gradient 梯度计算
        grad = tape.gradient(loss, trainable_variables)

        self.optimizer.apply_gradients(zip(grad, trainable_variables))

        return loss.numpy()
