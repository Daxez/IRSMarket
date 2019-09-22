"""
boxplot for the number of defaults in the system.
"""
from __future__ import division
import sys
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as pplot
import matplotlib.cm as clrs

from data.models import *
from analyse_hump import get_hump_error, get_aggregate_dist, get_hump_ks_error, get_hump_cumulative_value

def get_data(no_banks):
    query = """ 
    SELECT r.threshold, SUM(frequency)
    FROM run r
    INNER JOIN default_aggregate da ON r.aggregate_id = da.aggregate_id
    wHERE no_banks = :no_banks
    GROUP BY r.threshold, r.aggregate_id
    """
    session = get_session()
    data = session.execute(query, {'no_banks': no_banks}).fetchall()
    session.close()
    return data


if __name__ == '__main__':
    no_banks = 200
    data = get_data(no_banks)

    grouped = defaultdict(list)

    for (threshold,frequency) in data:
        grouped[threshold].append(float(frequency)/300000)
    
    boxdata = []
    keys = sorted(grouped.keys())
    for tpe in keys:
        boxdata.append(grouped[tpe])
    
    print len(boxdata)
    print len(keys)
    pplot.figure()

    pplot.boxplot(boxdata, labels=keys)

    pplot.title('Boxplot of the number of defaults per timestep for %d banks'%no_banks)
    pplot.xlabel('Threshold')
    pplot.ylabel('Total number of defaults per time step')

    pplot.show()


