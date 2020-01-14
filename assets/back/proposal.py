import tensorflow as tf


class Proposal(tf.keras.models.Model):
    def __init__(self):
        super(Proposal, self).__init__()

        # 将网络进行3x3卷积，得到一个共享网络层
        self.conv2d = tf.keras.layers.Conv2D(256, kernel_size=3, padding='SAME', activation=tf.nn.relu)

    def call(self, x):
        x = self.conv2d(x)
        return x
