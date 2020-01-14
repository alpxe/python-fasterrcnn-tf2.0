import tensorflow as tf

"""
使用已经训练好的vgg16网络提取图片特征 参考下载link:
https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5
如果自动下载 保存的路径 "/.keras/models/"
code::
    # include_top (True): 是否包括模型的输出层。如果您根据自己的问题拟合模型，则不需要这些(设置成False)。
    vgg16 = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=False)
    img = tf.keras.applications.vgg16.preprocess_input(img)
    print(img.shape) # 发现 MaxPooling2D 5次 下采样是 32
"""


def VGG16(vgg16_model_path):
    """
    仿 tf.keras.applications.vgg16.VGG16 的方法
    :param vgg16_model_path: 路径 vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5
    :return: models 16倍下采样
    """

    img_input = tf.keras.layers.Input(shape=(None, None, 3))

    # Block 1
    x = tf.keras.layers.Conv2D(64, (3, 3),
                               activation='relu',
                               padding='same',
                               name='block1_conv1')(img_input)
    x = tf.keras.layers.Conv2D(64, (3, 3),
                               activation='relu',
                               padding='same',
                               name='block1_conv2')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = tf.keras.layers.Conv2D(128, (3, 3),
                               activation='relu',
                               padding='same',
                               name='block2_conv1')(x)
    x = tf.keras.layers.Conv2D(128, (3, 3),
                               activation='relu',
                               padding='same',
                               name='block2_conv2')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = tf.keras.layers.Conv2D(256, (3, 3),
                               activation='relu',
                               padding='same',
                               name='block3_conv1')(x)
    x = tf.keras.layers.Conv2D(256, (3, 3),
                               activation='relu',
                               padding='same',
                               name='block3_conv2')(x)
    x = tf.keras.layers.Conv2D(256, (3, 3),
                               activation='relu',
                               padding='same',
                               name='block3_conv3')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = tf.keras.layers.Conv2D(512, (3, 3),
                               activation='relu',
                               padding='same',
                               name='block4_conv1')(x)
    x = tf.keras.layers.Conv2D(512, (3, 3),
                               activation='relu',
                               padding='same',
                               name='block4_conv2')(x)
    x = tf.keras.layers.Conv2D(512, (3, 3),
                               activation='relu',
                               padding='same',
                               name='block4_conv3')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = tf.keras.layers.Conv2D(512, (3, 3),
                               activation='relu',
                               padding='same',
                               name='block5_conv1')(x)
    x = tf.keras.layers.Conv2D(512, (3, 3),
                               activation='relu',
                               padding='same',
                               name='block5_conv2')(x)
    x = tf.keras.layers.Conv2D(512, (3, 3),
                               activation='relu',
                               padding='same',
                               name='block5_conv3')(x)

    # 注释掉第五次MaxPooling2D
    # x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    inputs = img_input
    model = tf.keras.models.Model(inputs, x, name='vgg16')

    model.load_weights(vgg16_model_path)

    return model
