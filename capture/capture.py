from abc import ABCMeta, abstractmethod

import capture

class ImgCapture(metaclass=ABCMeta):

    @abstractmethod
    def capture(self, output_file):
        pass

