from com.mvc.model.dataset_proxy import DatasetProxy
from com.mvc.model.label_proxy import LabelProxy
from com.mvc.view.application_mediator import ApplicationMediator
from puremvc.interfaces import INotification
from puremvc.patterns.command import SimpleCommand


class ApplicationStartupCommand(SimpleCommand):
    def execute(self, notification: INotification):
        self.facade.registerProxy(DatasetProxy())
        self.facade.registerProxy(LabelProxy())
        
        self.facade.registerMediator(ApplicationMediator(notification.getBody()))
        pass
