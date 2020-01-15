from com.mvc.app import App
from com.mvc.controller.notice_app import NoticeApp
from com.mvc.model.modellocator import ModelLocator
from com.mvc.view.component.neural_networks import NeuralNetworks
from com.mvc.view.network_mediator import NetworkMediator
from puremvc.interfaces import INotification
from puremvc.patterns.mediator import Mediator


class ApplicationMediator(Mediator):
    NAME = "ApplicationMediator"

    def __init__(self, viewComponent):
        super(ApplicationMediator, self).__init__(self.NAME, viewComponent)

    def listNotificationInterests(self):
        return [
            NoticeApp.INSTALL_MAIN_EVENT,
            NoticeApp.INSTALL_NETWORK_EVENT
        ]

    def handleNotification(self, notification: INotification):
        data = notification.getBody()

        if notification.getName() is NoticeApp.INSTALL_MAIN_EVENT:
            App().label_proxy.conversion()  # 创建 CSV 数据
            App().dataset_proxy.generate()  # CSV 转 训练集
            App().dataset_proxy.request_data(ModelLocator.train_recored_path)  # 获取训练数据

        elif notification.getName() is NoticeApp.INSTALL_NETWORK_EVENT:
            if not self.viewComponent.network:
                self.viewComponent.network = NeuralNetworks(data)
                self.facade.registerMediator(NetworkMediator(self.viewComponent.network))
