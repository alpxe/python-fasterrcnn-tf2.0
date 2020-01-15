from com.mvc.model.modellocator import ModelLocator
from puremvc.patterns.proxy import Proxy
import os
import glob
import simplejson as json
import pandas as pd


# Labelme to CSV
class LabelProxy(Proxy):
    NAME = "DataProxy"

    def __init__(self):
        super(LabelProxy, self).__init__(self.NAME)

    def conversion(self):
        if not os.path.exists(ModelLocator.csv_output_path):  # 如果文件不存在
            print("创建 CSV 文件")
            data = self.__label_to_csv(ModelLocator.label_image_path)
            data.to_csv(ModelLocator.csv_output_path, index=None)
            pass

    @staticmethod
    def __label_to_csv(path):
        json_list = []

        for f_url in glob.glob(path + '*.json'):
            with open(f_url, 'r') as f:
                temp = json.loads(f.read())
                for shape in temp['shapes']:
                    value = (
                        temp['imagePath'],
                        temp['imageWidth'],
                        temp['imageHeight'],
                        shape['label'],
                        shape['points'][0][0],
                        shape['points'][0][1],
                        shape['points'][1][0],
                        shape['points'][1][1]
                    )
                    json_list.append(value)

        column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']

        return pd.DataFrame(json_list, columns=column_name)
