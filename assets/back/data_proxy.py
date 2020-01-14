# from com.mvc.controller.notice_app import NoticeApp
# from puremvc.patterns.proxy import Proxy
# import tensorflow as tf
#
#
# class DataProxy(Proxy):
#     NAME = "DataProxy"
#
#     def __init__(self):
#         super(DataProxy, self).__init__(self.NAME)
#         pass
#
#     @staticmethod
#     def __format(record):
#         fs = {
#             "filename": tf.io.FixedLenFeature((), dtype=tf.string),
#             "image": tf.io.FixedLenFeature((), dtype=tf.string),
#             "width": tf.io.FixedLenFeature((), dtype=tf.int64),
#             "height": tf.io.FixedLenFeature((), dtype=tf.int64),
#             "xmin": tf.io.FixedLenFeature((), dtype=tf.float32),
#             "ymin": tf.io.FixedLenFeature((), dtype=tf.float32),
#             "xmax": tf.io.FixedLenFeature((), dtype=tf.float32),
#             "ymax": tf.io.FixedLenFeature((), dtype=tf.float32),
#             "label": tf.io.FixedLenFeature((), dtype=tf.int64),
#             "text": tf.io.FixedLenFeature((), dtype=tf.string)
#         }
#
#         return tf.io.parse_single_example(record, features=fs)
#
#     def request_data(self, path):
#         dataset = tf.data.TFRecordDataset(path)
#         dataset = dataset.repeat()
#         dataset = dataset.map(self.__format)
#         dataset = dataset.batch(1)
#         self.sendNotification(NoticeApp.INSTALL_NETWORK_EVENT, dataset)
