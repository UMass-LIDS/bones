import numpy as np


class BufferModel():
    def __init__(self, logger, max_buffer_level, num_seg, name):
        """
        Buffer model
        :param logger: information logger
        :param max_buffer_level: maximum buffer level, measured in ms
        :param num_seg: number of video segments
        :param name: name of the buffer
        """
        self.logger = logger
        self.max_level = max_buffer_level
        self.num_seg = num_seg
        self.name = name

        self.content = []  # store segment qualities
        self.history = np.zeros((num_seg, ), dtype=int)  # store history of decisions
        self.buffer_level = 0  # buffer level, measured in ms
        self.time_spent = 0  # time has been spent on the current segment

        # performance records
        self.idle_time = 0  # total idle time
        self.idle_event = 0  # idle event counts

    def consume(self, time):
        """
        Consume contents from the buffer
        :param time: time of content to consume
        :return:
        """
        if self.logger.is_verbose():
            self.logger.write("(consume) name {} time {:.2f}".format(self.name, time))
        if self.buffer_level == 0:
            # no content in buffer
            self.idle_time += time
            return

        if self.time_spent > 0:
            # finish processing the current segment
            _, _, segment_time = self.content[0]
            if time + self.time_spent < segment_time:
                # cannot use up the current segment
                self.time_spent += time
                self.buffer_level -= time
                return
            # use up the current segment
            time -= segment_time - self.time_spent
            self.buffer_level -= segment_time - self.time_spent
            self.content.pop(0)
            self.time_spent = 0

        # finished processing remaining segment, self.time_spent == 0 here
        while time > 0 and len(self.content) > 0:
            _, decision, segment_time = self.content[0]

            if time >= segment_time:
                # consume more than the current segment
                self.content.pop(0)
                self.buffer_level -= segment_time
                time -= segment_time
            else:
                self.time_spent = time
                self.buffer_level -= time
                time = 0

        if time > 0:
            # buffer drained
            self.idle_time += time
            self.idle_event += 1
            if self.logger.is_verbose():
                self.logger.write("(consume) name {} idle time {}".format(self.name, time))

        return

    def push(self, segment_index, decision, segment_time):
        """
        Push a new segment into the buffer
        :param segment_index: index of the segment
        :param decision: control decision of the segment
        :param segment_time: duration of the segment
        :return:
        """
        self.content.append((segment_index, decision, segment_time))
        self.history[segment_index] = decision
        self.buffer_level += segment_time

        # assert self.buffer_level < self.max_level + 500, "buffer overflow"
        if self.logger.is_verbose():
            self.logger.write("(push) name {} decision {} segment_time {:.2f}".format(
                self.name, decision, segment_time
            ))
        return

    def get_rest(self):
        """
        Get remaining time of the current segment
        :return: remaining time, None if no segment left
        """
        if len(self.content) > 0:
            _, _, segment_time = self.content[0]
            return segment_time - self.time_spent
        else:
            return None

def test_buffer_model():
    from logger import Logger
    buffer = BufferModel(Logger(), 10, 5)
    def report():
        print("level", buffer.get_level())

    buffer.push(0, 1, 1)
    report()
    buffer.consume(3)
    report()
    buffer.push(1, 2, 5)
    report()
    buffer.consume(3)
    report()
    buffer.consume(3)
    report()
    return


if __name__ == '__main__':
    test_buffer_model()

