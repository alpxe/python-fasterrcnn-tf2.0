from puremvc.patterns.proxy import Proxy


class DatasetProxy(Proxy):
    NAME = "DatasetProxy"

    def __init__(self):
        super(DatasetProxy, self).__init__(self.NAME)

    pass
