"""
Boxplot for the size of the gap in type 5 distributions
"""
from __future__ import division
import sys
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as pplot

from data.models import get_session

def get_data(no_banks):
    query = """ 
    SELECT DISTINCT r.aggregate_id, r.threshold, da.size
    FROM run r
    INNER JOIN default_aggregate da ON da.aggregate_id = r.aggregate_id
    WHERE r.no_banks = :no_banks
    ORDER BY r.aggregate_id, size
    """
    session = get_session()
    data = session.execute(query, {'no_banks': no_banks}).fetchall()
    session.close()
    return data


if __name__ == '__main__':
    no_banks = 200
    high_risk_threshold = 0.8
    data = get_data(no_banks)

    grouped = defaultdict(list)
    aggregate_to_threshold = dict()

    for (aggregate_id, threshold, size) in data:
        aggregate_to_threshold[aggregate_id] = threshold
        grouped[aggregate_id].append(size)
    
    gaps = defaultdict(list)

    for aggregate_id in grouped.keys():
        x = grouped[aggregate_id]

        if len([s for s in x if s > high_risk_threshold*no_banks]) > 1:
            gaps[aggregate_to_threshold[aggregate_id]].append(max([x[i+1]-x[i] for i in range(len(x)-1)]))

    keys = sorted(gaps.keys())
    
    print np.corrcoef(keys, [np.mean(gaps[k]) for k in keys])
    
    boxdata = []
    for tpe in keys:
        boxdata.append(gaps[tpe])
    
    pplot.figure()

    pplot.boxplot(boxdata, labels=keys)

    pplot.title('Boxplot of gap size for %d banks'%no_banks)
    pplot.xlabel('Threshold')
    pplot.ylabel('Size of gap')

    pplot.show()


