import numpy as np


class ModelLocator:
    # 标签路径
    label_image_path = "assets/train/image/"

    # csv输出路径
    csv_output_path = "assets/train/csv/train.csv"

    # 训练数据集 tfrecords
    train_recored_path = "assets/train/records/train.tfrecord"

    # 已经训练好的VGG16模型
    vgg16_model_path = "assets/h5/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"

    CLASSES = ('__background__',  # 0:背景标记必须有
               'jjy')

    stride = 16  # 下采样
    anchor_scales = 2 ** np.arange(5, 8)  # [8 16 32]
    anchor_ratios = [0.1, 1, 2]
    pass
