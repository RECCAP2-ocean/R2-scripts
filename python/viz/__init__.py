from . import maps
from .read_plotrc import read_plotrc


try:
    from matplotlib import pyplot as plt
    rc = read_plotrc()
    c = rc.colors
    fw1 = rc.figwidth.single
    fw2 = rc.figwidth.double
    plt.rcParams['figure.figsize'] = [fw2, 3]
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 150

except UnboundLocalError:
    print('rc not loaded')
    pass
