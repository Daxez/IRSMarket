"""Checking if we could create some collapse based on size. (hardcoded data in here)"""
from __future__ import division

import math
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict

from data.models import SwapModel, DefaultModel, RunModel, get_session
from heatmap import get_runs, get_defaults

if __name__ == '__main__':
    root_path = "./data_collapse"
    session = get_session()
    aggregate_ids = ['0772393d-881a-4895-9b8e-f9857a34aefc',
                     '67638748-c9d1-49ec-8b09-731a8a5cc383',
                     '98367a5d-4a69-444b-9011-148174cb94c3',
                     '5a846dd4-266e-414f-96ea-1b5fbf1c05c7',
                     '8ca7029f-7f6e-41e2-8775-2655a34b9fec']

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim((0.01,1))

    cnt = 0
    exponent = 2.55

    for aggregate_id in aggregate_ids:
        print "Going to get the runs"
        runs = get_runs(aggregate_id,session)

        all_defaults = []
        banks = 0

        for r in runs:
            print "Getting defaults for run"
            defs = get_defaults(r.run_id,session)

            for d in defs:
                all_defaults.append(d.size)

            banks = r.no_banks

        defaults = defaultdict(float)
        no_defs = len(all_defaults)
        for d in all_defaults:
            defaults[d] += pow(d,exponent)/no_defs

        del all_defaults

        print "Going to plot"


        lbl = "%d banks"%runs[0].no_banks
        plt.plot([k/banks for k in defaults.keys()],defaults.values(),label=lbl)

        cnt += 1
        print "%d of the %d done" %(cnt,len(aggregate_ids))

    plt.xlabel("s/sc")
    plt.ylabel("s^alpha * p(s|sc), alpha=%f"%exponent)
    plt.legend(loc=2)
    plt.savefig('./data_collapse/collapse.png')
    plt.show()
    plt.close()
