from . import maps
from .read_plotrc import read_plotrc


try:
    rc = read_plotrc()
    c = rc.colors
    fw1 = rc.figwidth.single
    fw2 = rc.figwidth.double
except UnboundLocalError:
    print('rc not loaded')
    pass
