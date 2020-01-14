# import pandas as pd
# import tensorflow as tf
# import os
# import sys
# import cv2
# from com.tool import dataset_util
# from com.core.base.singleton import Singleton
# from collections import namedtuple
# from com.mvc.model.modellocator import ModelLocator
#
#
# class Dataset(Singleton):
#
#     def __class_text_to_int(self, row_label):
#         """
#         文本标签标记分类
#         :param row_label:
#         :return:
#         """
#         if row_label == 'jjy':
#             return 1
#         else:
#             None
#         pass
#
#     def __split(self, df, fn):
#         data = namedtuple('data', "filename object")
#
#         gb = df.groupby(fn)  # [17 rows x 8 columns] 以名字分组
#
#         return [data(key, gb.get_group(key)) for key in gb.groups.keys()]
#
#     def __image_process(self, image):
#         """
#         修改图片尺寸
#         :param image:
#         :return:
#         """
#         h, w = image.shape[:2]
#
#         max_side = 512  # 最大边
#
#         pro = h / w
#         if pro > 0:
#             resize_h = max_side
#             resize_w = max_side * w / h
#         else:
#             resize_w = max_side
#             resize_h = max_side * h / w
#
#         return cv2.resize(image, (int(resize_w), int(resize_h)), interpolation=cv2.INTER_AREA)
#
#     def create_tf_example(self, item):
#         """
#         :param item: namedtuple('data', "filename object")
#         :return:
#         """
#         img_path = os.path.join(ModelLocator.label_path, item.filename)
#
#         # 读取图片
#         img = cv2.imread(img_path)
#         img = self.__image_process(img)  # 修正尺寸
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # bgr 转 rgb
#         encoded_jpg = cv2.imencode('.jpeg', img)[1].tostring()  # opencv  mat -> bytes
#
#         for index, row in item.object.iterrows():
#             width = row['width']
#             height = row['height']
#
#             # 坐标所在百分比
#             xmins = [row['xmin'] / width]
#             ymins = [row['ymin'] / height]
#
#             # 坐标所在百分比
#             xmaxs = [row['xmax'] / width]
#             ymaxs = [row['ymax'] / height]
#
#             classes_text = row['class']
#             classes = self.__class_text_to_int(row['class'])
#
#         height, width, depth = img.shape  # 获取缩放后的图片宽高深
#
#         fs = tf.train.Features(feature={
#             "filename": dataset_util.string_feature(item.filename),
#             "image": dataset_util.bytes_feature(encoded_jpg),
#             "width": dataset_util.int64_feature(width),
#             "height": dataset_util.int64_feature(height),
#             "xmin": dataset_util.float64_feature(xmins),
#             "ymin": dataset_util.float64_feature(ymins),
#             "xmax": dataset_util.float64_feature(xmaxs),
#             "ymax": dataset_util.float64_feature(ymaxs),
#             "label": dataset_util.int64_feature(classes),
#             "text": dataset_util.string_feature(classes_text),
#         })
#
#         return tf.train.Example(features=fs)
#
#     def generate(self, csv_path, train_path):
#         """
#         生成 tfrecords
#         :return:
#         """
#
#         if os.path.exists(train_path):  # 如果文件不存在
#             print("训练数据存在 \n-- end --")
#             return
#
#         examples = pd.read_csv(csv_path)
#
#         # namedtuple('data', "filename object") => filename='',object=''
#         grouped = self.__split(examples, 'filename')
#
#         with tf.io.TFRecordWriter(train_path) as writer:
#             for i, item in enumerate(grouped):
#                 example = self.create_tf_example(item)
#
#                 writer.write(example.SerializeToString())
#
#                 # 打印进度
#                 sys.stdout.write("\rWrite progress： {0:.2f}%".format((i + 1) / len(grouped) * 100))
#                 sys.stdout.flush()
