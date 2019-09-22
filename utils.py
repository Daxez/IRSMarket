"""Application utilities"""
from __future__ import division
import sys
import time
from datetime import datetime, timedelta
from uuid import UUID

# Class Progress was taken from StackOverflow
# http://stackoverflow.com/questions/15477122/fixing-a-python-progress-bar-in-command-prompt
# Thanks to Gianni Spear and chmulling
class Progress(object):
    def __init__(self, maxval):
        self._pct = 0
        self.maxval = maxval

    def update(self, value):
        pct = int((value / self.maxval) * 100.0)
        if self._pct != pct:
            self._pct = pct
            s = ((time.time()-self.start_time)/value)*(self.maxval-value)
            self.eta = datetime.now() + timedelta(seconds=s)
            self.display()

    def start(self):
        self.start_time = time.time()
        self.update(0)

    def finish(self):
        self.update(self.maxval)

    def display(self):
        sys.stdout.write("\r|%-73s| %3d%%" % ('#' * int(self._pct*.73), self._pct))
        sys.stdout.write(" ETA:%s"%self.eta.strftime('%I:%M'))
        sys.stdout.flush()

def is_valid_uuid(uuid_to_test, version=4):
    """
    Check if uuid_to_test is a valid UUID.

    Parameters
    ----------
    uuid_to_test : str
    version : {1, 2, 3, 4}

    Returns
    -------
    `True` if uuid_to_test is a valid UUID, otherwise `False`.

    Examples
    --------
    >>> is_valid_uuid('c9bf9e57-1685-4c89-bafb-ff5af830be8a')
    True
    >>> is_valid_uuid('c9bf9e58')
    False
    """
    try:
        uuid_obj = UUID(uuid_to_test, version=version)
    except:
        return False

    return str(uuid_obj) == uuid_to_test

if __name__ == '__main__':
    print is_valid_uuid('56410d36-b0ed-4740-8a16-55224986d12f')
