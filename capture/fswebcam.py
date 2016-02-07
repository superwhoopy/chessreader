import subprocess
import utils
import capture


class Fswebcam(capture.ImgCapture):
    '''Capture an image from a video device using `fswebcam`

    See `fswebcam github <https://github.com/fsphil/fswebcam>`_.

    Attributes:
        resolution (str): Final image resolution e.g. "1920x1080"
        frames_per_capture (int): Number of frames to be captured and merged
            into the final output file. The more frames, the less noisy the
            image - but the capture will take a bit longer.
        default_output (str): Default output file name
        stdout (fileno): `stdout` stream redirection; by default goes to
            `/dev/null` - careful, `fswebcam` prints out everything on
            `stderr`...
        stderr (fileno): `stderr` stream redirection, goes to `/dev/null` by
            default.

    Example:
        Capture and merge 10 HD frames into a single jpeg called
        `my_output_file.jpg`:

        >>> cap = Fswebcam()
        >>> cap.resolution='1280x720'
        >>> cap.frames_per_capture=10
        >>> cap.capture('my_output_file.jpg')
    '''

    BIN            = 'fswebcam'
    OPT_VERSION    = '--version'
    OPT_HELP       = '--help'
    OPT_RESOLUTION = '--resolution="{}"'
    OPT_FRAMES     = '--frames={}'
    OPT_OUTPUT     = '--save={}'
    OPT_PALETTE    = '--palette={}'
    OPT_SKIP       = '--skip={}'
    OPT_ROTATE     = '--rotate={}'

    DEFAULT_TIMEOUT = 15

    @staticmethod
    def _check_bin():
        '''Make sure that fswebcam exists and ask for its version'''
        cmdline = [Fswebcam.BIN, Fswebcam.OPT_VERSION]
        utils.log.debug('calling ' + ' '.join(cmdline))

        try:
            subprocess.check_call(cmdline, timeout=Fswebcam.DEFAULT_TIMEOUT,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as caught:
            if caught.returncode != 255:
                raise

    def __init__(self):
        self._check_bin()

        # DEFAULT VALUES
        self.resolution          = '1024x768'
        self.rotate              = 0
        self.frames_per_capture  = 10
        self.skip_before_capture = 3
        self.default_output      = 'capture.jpg'

        self.stdout              = subprocess.DEVNULL
        self.stderr              = subprocess.DEVNULL

    def _cmdline(self):
        return [
                self.BIN,
                '--no-banner',
                self.OPT_RESOLUTION.format(self.resolution),
                self.OPT_FRAMES.format(self.frames_per_capture),
                self.OPT_SKIP.format(self.skip_before_capture),
                self.OPT_ROTATE.format(self.rotate),
              ]

    def capture(self, output_file=None):
        '''Capture one frame

        Args:
            output_file (str): name of the output JPG file
        '''
        output_file = output_file or self.default_output
        args_list = [ self.OPT_OUTPUT.format(output_file) ]

        cmdline = self._cmdline() + args_list
        utils.log.debug('calling "' + ' '.join(cmdline) + '"')
        subprocess.check_call(cmdline, timeout=self.DEFAULT_TIMEOUT,
                stdout=self.stdout, stderr=self.stderr)

        return output_file

