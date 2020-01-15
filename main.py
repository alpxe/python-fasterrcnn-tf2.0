from com.mvc.controller.application_facade import ApplicationFacade
from com.mvc.controller.notice_app import NoticeApp


class Main:
    network = None

    def __init__(self):
        self.__initMVC()
        pass

    def __initMVC(self):
        ApplicationFacade.getInstance(ApplicationFacade.NAME).startup(self)
        ApplicationFacade.getInstance(ApplicationFacade.NAME).sendNotification(NoticeApp.INSTALL_MAIN_EVENT)
        pass


if __name__ == "__main__":
    Main()
