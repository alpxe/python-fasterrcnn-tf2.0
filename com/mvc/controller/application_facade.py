from com.mvc.controller.application_startup_command import ApplicationStartupCommand
from puremvc.patterns.facade import Facade


class ApplicationFacade(Facade):
    NAME = "FASTER_RCNN_MODULE"
    STARTUP = "STARTUP_EVENT"

    def __init__(self, key):
        super(ApplicationFacade, self).__init__(key)

    def initializeController(self):
        super(ApplicationFacade, self).initializeController()
        self.registerCommand(self.STARTUP, ApplicationStartupCommand)

    def startup(self, app):
        self.sendNotification(self.STARTUP, app)
