from abc import ABCMeta, abstractmethod

class ImgCapture(metaclass=ABCMeta):

    @abstractmethod
    def capture(self, output_file):
        pass

