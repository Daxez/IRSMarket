"""Graphs the mean and stdev

old
"""
from __future__ import division

import pickle
import sys
import matplotlib.pyplot as pplot
import numpy as np

from utils import is_valid_uuid

if __name__ == '__main__':
    if len(sys.argv)>1 and is_valid_uuid(sys.argv[1]):
        aggregate_id = sys.argv[1]
    else:
        aggregate_id = '6e1dcc6f-4288-4e1e-b475-05809888748b'

    path = './simulation_data/%s/balances.bin'%aggregate_id

    means = None
    stds = None
    nets = None

    with open(path,'rb') as fp:
        means = np.array(pickle.load(fp))
        stds = np.array(pickle.load(fp))
        nets = np.array(pickle.load(fp))

    rng = np.arange(len(means))

    fig = pplot.figure()
    ax = fig.add_subplot(211)
    ax.plot(rng,means+stds,'r-')
    ax.plot(rng,means-stds,'r-')
    ax.plot(rng,means,'b-')

    ax = fig.add_subplot(212)
    ax.plot(rng,nets,'b-')

    pplot.show()
