

class SessionModel():
    def __init__(self, logger):
        """
        Decision session manager
        There are two timelines for download buffer and enhance buffer respectively
        Each timeline alternatively enters processing session and waiting session
        There are four sessions in total. At the beginning of each session, we:
            0: finish waiting, start downloading
            1: finish downloading, start waiting
            2: finish waiting, start enhancing
            3: finish enhancing, start waiting
        :param logger: information logger
        """
        self.DOWNLOAD_START = 0
        self.DOWNLOAD_FINISH = 1
        self.ENHANCE_START = 2
        self.ENHANCE_FINISH = 3

        self.logger = logger

        self.is_downloading = False  # True: downloading, False: waiting
        self.cd_download = 0  # download session countdown

        self.is_enhancing = False  # True: enhancing, False: waiting
        self.cd_enhance = 0  # enhance session countdown

    def next(self):
        """
        Enter next session
        :return: index of NEXT session, remaining time of CURRENT session
        """
        # enter the session with the least remaining time
        if self.cd_download < self.cd_enhance:
            time_session = self.cd_download
            self.cd_enhance -= time_session
            self.cd_download = 0
            if not self.is_downloading:
                # finish waiting, start downloading
                if self.logger.is_verbose():
                    self.logger.write("(session) download_start")
                return self.DOWNLOAD_START, time_session
            else:
                # finish downloading, start waiting
                if self.logger.is_verbose():
                    self.logger.write("(session) download_finish")
                return self.DOWNLOAD_FINISH, time_session
        else:  # self.cd_enhance < self.cd_download
            time_session = self.cd_download
            self.cd_download -= time_session
            self.cd_enhance = 0
            if not self.is_enhancing:
                # finish waiting, start enhancing
                if self.logger.is_verbose():
                    self.logger.write("(session) enhance_start")
                return self.ENHANCE_START, time_session
            else:
                # finish enhancing, start waiting
                if self.logger.is_verbose():
                    self.logger.write("(session) enhance_finish")
                return self.ENHANCE_FINISH, time_session

    def set_download(self, is_doing, cd):
        self.is_downloading = is_doing
        self.cd_download = cd
        return

    def set_enhance(self, is_doing, cd):
        self.is_enhancing = is_doing
        self.cd_enhance = cd
        return



