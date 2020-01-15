from com.mvc.app import App


class NeuralNetworks:
    def __init__(self, dataset):
        print("NeuralNetworks")
        # self.vgg16 = VGG16(ModelLocator.vgg16_model_path)
        for step, item in enumerate(dataset):
            self.__training(item)
            break

    def __training(self, data):
        image, image_width, image_height, gt_boxes = App().dataset_proxy.analytical(data)
        pass
