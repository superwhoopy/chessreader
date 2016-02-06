from nose.tools import eq_, ok_
import os, filecmp

from capture import Mock

class ErrMsg:
    FILE_NOT_FOUND = \
            'file not found: "{}"'
    FILE_DIFFER = \
            'files should be identical: "{}" and "{}"'


def test_mock():
    '''capture interface mock'''
    files_list = [ 'tests/pictures/board-{}.jpg'.format(idx) \
                   for idx in range(0,17) ]
    cap = Mock(files_list)

    for idx in range(0,17):
        eq_(cap.capture(), files_list[idx])

    # test file copy
    copy_file = './output_file.jpg'
    cap.capture(copy_file)
    ok_(os.path.exists(copy_file), ErrMsg.FILE_NOT_FOUND.format(copy_file))
    ok_(filecmp.cmp(copy_file, files_list[0]),
            ErrMsg.FILE_DIFFER.format(copy_file, files_list[0]))
    # remove this useless copy
    os.remove(copy_file)
