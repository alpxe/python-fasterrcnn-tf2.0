from com.mvc.controller.notice_app import NoticeApp
from com.mvc.model.label_proxy import LabelProxy
from puremvc.interfaces import INotification
from puremvc.patterns.mediator import Mediator


class ApplicationMediator(Mediator):
    NAME = "ApplicationMediator"

    def __init__(self, viewComponent):
        super(ApplicationMediator, self).__init__(self.NAME, viewComponent)

    def listNotificationInterests(self):
        return [
            NoticeApp.INSTALL_MAIN_EVENT
        ]

    def handleNotification(self, notification: INotification):
        data = notification.getBody()

        if notification.getName() is NoticeApp.INSTALL_MAIN_EVENT:
            self.facade.retrieveProxy(LabelProxy.NAME).conversion()

            pass
        # elif notification.getName() is NoticeApp.INSTALL_NETWORK_EVENT:
        # if not self.getViewComponent().net:
        #     self.getViewComponent().net = NetWork(data)
        #     self.facade.registerMediator(NetworkMediator(self.getViewComponent().net))

    def getViewComponent(self):
        return self.viewComponent
