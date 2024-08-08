import warnings
from numba import NumbaPerformanceWarning
warnings.resetwarnings()
warnings.simplefilter('ignore', NumbaPerformanceWarning)

from .model import LaNoLem
from .utils import *