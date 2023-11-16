import numpy as np
import matplotlib.pyplot as plt
from plog import Log

def test_sample():
    lg = Log([0.1, 0.4, 0.5], [0, 5, 7.5], [5, 7.5, 10.], 'random')
    val = lg.sample(3.)
    print(val)
    assert val==0.1, "sampling function is not working"

if __name__=='__main__':
    test_sample()