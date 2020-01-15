from puremvc.interfaces import INotification
from puremvc.patterns.mediator import Mediator


class NetworkMediator(Mediator):
    NAME = "NetworkMediator"

    def __init__(self, viewComponent):
        super(NetworkMediator, self).__init__(self.NAME, viewComponent)

    def listNotificationInterests(self):
        return []

    def handleNotification(self, notification: INotification):
        pass
