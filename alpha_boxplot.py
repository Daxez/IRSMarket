"""
    File for generating a boxplot graph on the values of alpha
    run as is
"""

from __future__ import division
import sys
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as pplot
import matplotlib.cm as clrs

from data.models import *
from analyse_hump import get_hump_error, get_aggregate_dist, get_hump_ks_error, get_hump_cumulative_value

def get_data():
    query = """ 
    SELECT type_id, aggregate_powerlaw.alpha FROM aggregate_type
    INNER JOIN aggregate_powerlaw ON aggregate_powerlaw.aggregate_id = aggregate_type.aggregate_id
    wHERE aggregate_powerlaw.alpha < 1000
    """
    session = get_session()
    data = session.execute(query).fetchall()
    session.close()
    return data


if __name__ == '__main__':
    data = get_data()

    grouped = defaultdict(list)

    for (type,alpha) in data:
        grouped[type].append(alpha)
    
    boxdata = []
    for tpe in grouped.keys():
        boxdata.append(grouped[tpe])
    
    pplot.figure()

    pplot.boxplot(boxdata, labels=['1','2','3','4','5'])

    pplot.title('Boxplot of alpha for distribution types')
    pplot.xlabel('Distribution type')
    pplot.ylabel('Alpha')

    pplot.show()


