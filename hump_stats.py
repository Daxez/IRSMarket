"""Script for measuring humps and storing some properties of then in the database"""
from __future__ import division

import math
import pickle
import time

import numpy as np

from sqlalchemy import func
from sqlalchemy.orm import load_only
from data.models import *
from scipy.special import zeta

from hump_detector import HumpEstimator

if __name__ == '__main__':
    session = get_session()
    runs = session.query(RunModel).all()
    session.close()

    print "Got runs"

    #aggregate_ids = list(set([r.aggregate_id for r in runs]))
    aggregate_ids = ["b601e873-2e3e-40ea-b466-fb032b838699","3094a622-4d0c-400e-a4ec-52a52b11d3d2",
                     "095874a7-2dee-44ec-8036-4ee35e1bbaea","c53158f2-771c-48ba-ad0f-ce4305d41d93"]

    res = []
    no_aggs = len(aggregate_ids)

    print "Start analyzing"

    for (i,aggregate_id) in enumerate(aggregate_ids):
        start = time.time()
        with HumpEstimator(aggregate_id, True) as he:
            he.get_default_counts()

            error = he.calculate_powerlaw_error()
            pli = he.last_pl_index
            alpha = he.alpha

            res.append(AggregatePowerlaw(aggregate_id, pli, alpha, error))

        print "Done %d of %d"%(i,no_aggs)
        print "Did this round in %d seconds"%(time.time()-start)

    #with open('simulation_data/dta_types.bin','wb') as fp:
    #    pickle.dump(res,fp)

    session = get_session()
    session.bulk_save_objects(res)
    session.commit()
    session.close()
