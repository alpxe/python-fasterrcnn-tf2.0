from com.core.base.singleton import Singleton
from com.mvc.controller import application_facade as facade
from com.mvc.model import dataset_proxy as dataset
from com.mvc.model import label_proxy as label


class App(Singleton):

    @property
    def facade(self):
        return facade.ApplicationFacade.getInstance(facade.ApplicationFacade.NAME)

    @property
    def dataset_proxy(self) -> dataset.DatasetProxy:
        return self.facade.retrieveProxy(dataset.DatasetProxy.NAME)

    @property
    def label_proxy(self) -> label.LabelProxy:
        return self.facade.retrieveProxy(label.LabelProxy.NAME)
