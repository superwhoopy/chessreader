import itertools, shutil, os

import capture, utils

class Mock(capture.ImgCapture):
    '''Mock image capture class: mimics a webcam from existing image files

    When instantiating `Mock`, simply provide a list of image files that will be
    returned in cycle everytime you call the `capture()` method. If you ask for
    a specific `output_file`, the original file will be copied to this location.

    Args:
        files_list (list): List of images to be returned in cycle by `capture()`

    Example:
        Provided 3 image files `a.jpg`, `b.jpg` and `c.jpg` in the current
        directory:

        >>> cap = Mock(['a.jpg', 'b.jpg', 'c.jpg'])
        >>> cap.capture()
        'a.jpg'
        >>> cap.capture('output.jpg') # copies b.jpg to output.jpg
        'output.jpg'
        >>> cap.capture()
        'c.jpg'
        >>> cap.capture()
        'a.jpg'
    '''

    def __init__(self, files_list):

        def _check_file(file_path):
            '''Check that a file exists and returns its normalized path

            Args:
                file_path: path to the file to be tested
            Raises:
                OSError: raised if `file_path` does not exist in the filesystem
            Returns:
                str: normalized version of `file_path`
            '''
            file_path = os.path.normpath(file_path)
            if not os.path.exists(file_path):
                raise OSError(None, "file not found", file_path)
            return file_path

        self.files_list = [ _check_file(f) for f in files_list ]
        self.iterator = itertools.cycle(files_list)

    def capture(self, output_file=None):
        next_file = next(self.iterator)
        if output_file:
            output_file = os.path.normpath(output_file)
            shutil.copyfile(next_file, output_file)
            next_file = output_file
        utils.log.debug("Sending `{0}`".format(os.path.basename(next_file)))
        return next_file

