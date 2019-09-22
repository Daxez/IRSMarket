"""OLD Script for measuring humps"""
from __future__ import division

import math
import numpy as np
import powerlaw

import matplotlib.pyplot as pplot

from sqlalchemy import func
from sqlalchemy.orm import load_only
from data.models import *
from scipy.special import zeta


class HumpEstimator:

    def __init__(self, aggregate_id,manual=False):
        self.aggregate_id = aggregate_id
        self.defaults = []
        self.xmax = None
        self.last_pl_index = None
        self.results = None
        self.manual = manual

    def get_default_counts(self):
        session = get_session()

        self.defaults = session.execute("""SELECT size, frequency
                                  FROM default_aggregate
                                  WHERE aggregate_id = :aggregate_id
                                  ORDER BY size""",
                                  {'aggregate_id':self.aggregate_id}).fetchall()

        self.n = sum([a[1] for a in self.defaults])

        self.run = session.query(RunModel)\
                           .filter(RunModel.aggregate_id == self.aggregate_id)\
                           .first()
        self.xmax = self.run.no_banks

    def plot_simple(self):
        if(len(self.defaults) == 0):
            self.get_default_counts()

        fig = pplot.figure()
        ax = fig.add_subplot(111)
        ax.set_yscale('log')
        ax.set_xscale('log')

        ax.scatter([x[0] for x in self.defaults],[y[1] for y in self.defaults])
        pplot.show()
        pplot.close('all')


    def get_power_law_dataset(self):
        if(len(self.defaults) == 0):
            self.get_default_counts()

        if(self.last_pl_index != None):
            return self.defaults[:self.last_pl_index]

        if not self.manual:
            prev = self.defaults[0]
            for i in xrange(1,len(self.defaults)):
                if prev[1] == self.defaults[i][1]:
                    self.last_pl_index = i
                    return self.defaults[:i]
                prev = self.defaults[i]
        else:
            self.plot_simple()

            inp = raw_input("Please enter the size of the last powerlaw\n")
            while(not inp.isdigit()):
                inp = raw_input("Not an int, try again\n")

            val = int(inp)
            if(val == 0):
                return self.defaults

            for (i,(s,f)) in enumerate(self.defaults):
                if(s == val):
                    self.last_pl_index = i+1
                    return self.defaults[:self.last_pl_index]
                elif(s > val):
                    self.last_pl_index = i
                    return self.defaults[:self.last_pl_index]
        return self.defaults

    def get_power_law(self):
        defaults = self.get_power_law_dataset()

        ds = []
        for size,n in defaults:
            for i in xrange(n):
                ds.append(size)

        self.results = powerlaw.Fit(ds, discrete=True, estimate_discrete=False, xmin=1)
        self.alpha = self.results.alpha
        self.c = 1/(zeta(self.alpha,1))

        return self.results.alpha

    def test_power_law(self):
        dist1 = 'power_law'
        dist2 = 'lognormal'

        if(self.results == None):
            self.get_power_law()

        R, p = self.results.distribution_compare(dist1,dist2)

        print 'R -> likelyhood for %s against %s and p is the significance of the likelyhood'%(dist1,dist2)
        print 'R:%f, p:%f'%(R,p)

        return R,p

    def plot(self):
        self.get_power_law()
        ds = self.defaults

        fig = pplot.figure()
        ax = fig.add_subplot(111)

        ax.set_xscale('log')
        ax.set_yscale('log')

        n = sum([a[1] for a in ds])

        ax.scatter([a[0] for a in ds], [a[1]/n for a in ds],label='data')
        ax.plot(xrange(1,self.xmax), [self.c*a**(-self.alpha) for a in xrange(1,self.xmax)],'r-',label='Power law')
        self.results.lognormal.plot_pdf(ax=ax,color='b',linestyle='-.',label='Lognormal')

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc=3)
        ax.set_title("Avalanche Size Distribution, Alpha=%f"%self.results.alpha)

        pplot.show()

    def calculate_powerlaw_error(self):
        if(self.results == None):
            self.get_power_law()

        erfunc = lambda a: abs((a[1]/self.n) - (self.c*a[0]**(-self.alpha)))

        if self.last_pl_index != None and self.last_pl_index > 0:
            ds = self.defaults[self.last_pl_index:]
        else:
            ds = []

        self.error = sum([erfunc(a) for a in ds])
        return self.error

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del self.defaults
        del self.results


if __name__ == '__main__':
    aggregate_id = 'd2704589-aea4-486e-a282-2961859f1b71'
    a = HumpEstimator(aggregate_id, True)
    a.test_power_law()
    print a.calculate_powerlaw_error()
    a.plot()
