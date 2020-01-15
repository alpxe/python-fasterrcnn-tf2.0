from com.mvc.controller.notice_app import NoticeApp
from com.mvc.model.modellocator import ModelLocator
from puremvc.patterns.proxy import Proxy
from collections import namedtuple
import os
import pandas as pd
import tensorflow as tf
import sys
import cv2


class DatasetProxy(Proxy):
    NAME = "DatasetProxy"

    def __init__(self):
        super(DatasetProxy, self).__init__(self.NAME)

    @staticmethod
    def analytical(data):
        image = tf.cast(tf.image.decode_image(data["image"][0]), dtype=tf.float32) / 225.  # 归一化
        image = tf.expand_dims(image, 0)

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

    def request_data(self, path):
        """
        返回数据集
        :param path: 数据集路径
        :return:
        """

        dataset = tf.data.TFRecordDataset(path)
        dataset = dataset.repeat()
        dataset = dataset.map(self.__format)
        dataset = dataset.batch(1)
        self.sendNotification(NoticeApp.INSTALL_NETWORK_EVENT, dataset)

    @staticmethod
    def __format(record):
        fs = {
            "filename": tf.io.FixedLenFeature((), dtype=tf.string),
            "image": tf.io.FixedLenFeature((), dtype=tf.string),
            "width": tf.io.FixedLenFeature((), dtype=tf.int64),
            "height": tf.io.FixedLenFeature((), dtype=tf.int64),
            "xmin": tf.io.FixedLenFeature((), dtype=tf.float32),
            "ymin": tf.io.FixedLenFeature((), dtype=tf.float32),
            "xmax": tf.io.FixedLenFeature((), dtype=tf.float32),
            "ymax": tf.io.FixedLenFeature((), dtype=tf.float32),
            "label": tf.io.FixedLenFeature((), dtype=tf.int64),
            "text": tf.io.FixedLenFeature((), dtype=tf.string)
        }

        return tf.io.parse_single_example(record, features=fs)

    def generate(self):
        """
        生成 tfrecords
        """

        if os.path.exists(ModelLocator.train_recored_path):  # 如果文件不存在
            print("训练数据存在 \n-- end --")
            return

        # 读取csv数据
        examples = pd.read_csv(ModelLocator.csv_output_path)

        # namedtuple('data', "filename object") => filename='',object=''
        grouped = self.__split(examples, 'filename')

        with tf.io.TFRecordWriter(ModelLocator.train_recored_path) as writer:
            for i, item in enumerate(grouped):
                example = self.__create_tf_example(item)

                writer.write(example.SerializeToString())

                # 打印进度
                sys.stdout.write("\rWrite progress： {0:.2f}%".format((i + 1) / len(grouped) * 100))
                sys.stdout.flush()
        pass

    def __create_tf_example(self, item):
        """
        :param item: namedtuple('data', "filename object")
        :return:
        """
        img_path = os.path.join(ModelLocator.label_image_path, item.filename)

        # 读取图片
        img = cv2.imread(img_path)
        img = self.__image_process(img)  # 修正尺寸
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # bgr 转 rgb
        encoded_jpg = cv2.imencode('.jpeg', img)[1].tostring()  # opencv  mat -> bytes

        for index, row in item.object.iterrows():
            width = row['width']
            height = row['height']

            # 坐标所在百分比
            xmins = [row['xmin'] / width]
            ymins = [row['ymin'] / height]

            # 坐标所在百分比
            xmaxs = [row['xmax'] / width]
            ymaxs = [row['ymax'] / height]

            classes_text = row['class']
            classes = self.__class_text_to_int(row['class'])

        height, width, depth = img.shape  # 获取缩放后的图片宽高深

        fs = tf.train.Features(feature={
            "filename": self.dataset_util.string_feature(item.filename),
            "image": self.dataset_util.bytes_feature(encoded_jpg),
            "width": self.dataset_util.int64_feature(width),
            "height": self.dataset_util.int64_feature(height),
            "xmin": self.dataset_util.float64_feature(xmins),
            "ymin": self.dataset_util.float64_feature(ymins),
            "xmax": self.dataset_util.float64_feature(xmaxs),
            "ymax": self.dataset_util.float64_feature(ymaxs),
            "label": self.dataset_util.int64_feature(classes),
            "text": self.dataset_util.string_feature(classes_text),
        })

        return tf.train.Example(features=fs)

    @staticmethod
    def __image_process(image):
        """
        修改图片尺寸
        :param image:
        :return:
        """
        h, w = image.shape[:2]

        max_side = 512  # 最大边

        pro = h / w
        if pro > 0:
            resize_h = max_side
            resize_w = max_side * w / h
        else:
            resize_w = max_side
            resize_h = max_side * h / w

        return cv2.resize(image, (int(resize_w), int(resize_h)), interpolation=cv2.INTER_AREA)

    @staticmethod
    def __class_text_to_int(row_label):
        """
        文本标签标记分类
        :param row_label:
        :return:
        """
        if row_label == 'jjy':
            return 1
        else:
            None
        pass

    @staticmethod
    def __split(df, fn):
        data = namedtuple('data', "filename object")
        gb = df.groupby(fn)  # [17 rows x 8 columns] 以名字分组

        return [data(key, gb.get_group(key)) for key in gb.groups.keys()]

    @staticmethod
    def int64_feature(value):
        if type(value) is int:
            value = [value]

        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    @staticmethod
    def float64_feature(value):
        if type(value) is float:
            value = [value]

        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    @staticmethod
    def bytes_feature(value):
        if type(value) is bytes:
            value = [value]
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    @staticmethod
    def string_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode('utf-8')]))
