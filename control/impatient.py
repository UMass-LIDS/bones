import math


def insert_impatient_enhancement(self):
    """
    Insert impatient enhancement to a simulator object
    :param self:
    :return:
    """
    # choose enhancement methods
    self.action_enh_list = []
    for idx_bitrate in range(self.enhance_time.shape[0]):
        best_quality = 0
        best_action = 0
        for idx_action in range(self.enhance_time.shape[1]):
            if self.enhance_time[idx_bitrate][idx_action] < self.seg_time:
                if self.vmaf_enh_avg[idx_bitrate][idx_action] > best_quality:
                    best_quality = self.vmaf_enh_avg[idx_bitrate][idx_action]
                    best_action = idx_action
        self.action_enh_list.append(best_action)

    # overwrite control functions
    self.control_download_finish = control_download_finish.__get__(self)
    self.control_enhance_start = control_enhance_start.__get__(self)
    self.control_enhance_finish = control_enhance_finish.__get__(self)
    return


def control_download_finish(self):
    """
    Control action when download finishes
    :return:
    """
    # awake the enhancement buffer if it's sleeping
    if self.buffer_enhance.buffer_level == 0:
        self.session.set_enhance(False, 0)

    buffer_level = self.buffer_download.buffer_level
    max_level = self.buffer_download.max_level
    time_wait = buffer_level + self.seg_time - max_level
    time_wait = max(0, time_wait)
    self.session.set_download(False, time_wait)
    return


def control_enhance_start(self):
    """
    Control action when enhancement starts
    :return:
    """
    # sleep if the next segment hasn't been downloaded
    if self.idx_enh >= self.idx_down:
        self.session.set_enhance(False, math.inf)
        return

    # choose the enhancement action for the download bitrate
    action_down = self.buffer_download.history[self.idx_enh]
    action_enh = self.action_enh_list[action_down]

    # enhancement must be finished before playback
    buff_down = self.buffer_download.buffer_level
    buff_enh = self.buffer_enhance.buffer_level
    if buff_down - buff_enh - self.seg_time > self.enhance_time[action_down][action_enh]:
        self.action_enh = action_enh
    else:
        self.action_enh = 0

    # push the corresponding computation task into the enhancement buffer
    if self.action_enh != 0:
        self.buffer_enhance.push(segment_index=self.idx_enh, decision=self.action_enh,
                                 segment_time=self.enhance_time[action_down][self.action_enh])

    self.idx_enh += 1

    if self.logger.is_verbose():
        self.logger.write("(control) enhance {}".format(self.action_enh))
    return


def control_enhance_finish(self):
    """
    Control action when enhancement finishes
    :return:
    """
    return