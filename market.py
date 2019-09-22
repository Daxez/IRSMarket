"""
OLD
"""
"""Market: general process"""
# Author: Dexter Drupsteen
from __future__ import division

import itertools
import logging
import numpy as np
from math import fabs

from collections import defaultdict
from basegraphs import Graph
from bank import Bank
from irs import IRS
from random import shuffle


class Market(Graph):
    def __init__(self,modelconfig,datacontainer,step_callback,seed=None):
        super(Market,self).__init__()

        self.no_steps = modelconfig['no_steps']
        self.irs_threshold = modelconfig['irs_threshold']
        self.max_irs_value = modelconfig['max_irs_value']
        self.max_tenure = modelconfig['max_tenure']
        self.no_banks = modelconfig['no_banks']
        self.sigma = modelconfig['sigma']
        self.__threshold__ = modelconfig['threshold']

        self.datacontainer = datacontainer
        self.step_callback = step_callback
        self.__time__ = 0

        self.seed = seed
        if(self.seed == None):
            # Max seed value
            self.seed = np.random.randint(4294967295)
        self.generator = np.random.RandomState(self.seed)

        self.datacontainer.register_seed(self.seed)

        for i in xrange(self.no_banks):
            b = Bank(sigma=self.sigma)
            self.add_node(b)

    @property
    def banks(self):
        for n in self.node:
            yield n

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for irs in self.get_edges():
            irs.remove()
            del irs

        banks = [b for b in self.banks]
        for bank in banks:
            self.datacontainer.register_bank(bank.name,bank.seed)
            bank.graph = None
            self.remove_node(bank)
            del bank

        self.datacontainer = None
        self.config = None

    @property
    def time(self):
        return self.__time__

    @property
    def threshold(self):
        return self.__threshold__

    def run(self):
        for i in xrange(self.no_steps):
            self.time_step()

    def istep(self,obj):
        obj.time_step()

    def icheck_balance(self,obj):
        obj.check_balance()

    def iregister_balance(self,obj):
        obj.register_balance()

    def time_step(self):
        map(self.istep,self.get_edges())
        map(self.istep,self.banks)

        self.create_swaps()

        map(self.icheck_balance,self.banks)
        map(self.iregister_balance,self.banks)

        self.__time__ += 1
        if(self.step_callback != None):
            self.step_callback(self.time)

    def create_swaps(self):
        self.banks.sort(key=lambda x: x.balance)
        most_floated = self.banks[0]
        most_fixed = self.banks[-1]

        if(most_floated.balance < 0 and most_fixed.balance > 0):
            self.create_swap(most_floated,most_fixed)

    def get_tenure_time(self,irs_value):
        return self.max_tenure

    def create_swap(self,fl,fx):
        irs_val = min(self.max_irs_value,min(abs(fl.balance),fx.balance))
        tenure = self.max_tenure
        self.add_edge(fx,fl,IRS(irs_val,tenure,self.time))

        fx.set_dirty()
        fl.set_dirty()

    def icreate_swap(self,fs):
        self.create_swap(fs[0],fs[1])

#
# Market with chances for IRS creation instead of the default one swap per dt
#
class ShuffleIRSMarket(Market):

    def create_swaps(self):
        floats = [b for b in self.banks if b.balance < 0 and fabs(b.balance) > self.irs_threshold]
        fixs = [b for b in self.banks if b.balance > 0 and b.balance > self.irs_threshold]

        self.generator.shuffle(floats)
        self.generator.shuffle(fixs)

        map(self.icreate_swap, itertools.izip(floats,fixs))

        #for (fl,fx) in itertools.izip(floats,fixs):
        #    self.create_swap(fl,fx)

class ConstShuffleIRSMarket(ShuffleIRSMarket):

    def create_swaps(self):
        floats = [b for b in self.banks if b.balance < 0 and fabs(b.balance) > self.irs_threshold]
        fixs = [b for b in self.banks if b.balance > 0 and b.balance > self.irs_threshold]

        self.generator.shuffle(floats)
        self.generator.shuffle(fixs)

        map(self.icreate_swap, itertools.izip_longest(floats,fixs))

    def icreate_swap(self,fs):
        if(fs[0] == None):
            fs[1].register_failed_irs()
            return
        if(fs[1] == None):
            fs[0].register_failed_irs()
            return

        self.create_swap(fs[0],fs[1])
        fs[0].register_success_irs()
        fs[1].register_success_irs()

    def create_swap(self,fl,fx):
        irs_val = self.irs_threshold
        tenure = self.max_tenure
        self.add_edge(fx,fl,IRS(irs_val,tenure,self.time))

        fx.set_dirty()
        fl.set_dirty()

#
# Market in which the agents most in need get swaps with eachother
#
class SortedIRSMarket(Market):

    def create_swaps(self):
        floats = [b for b in self.banks if b.balance < 0 and fabs(b.balance) > self.irs_threshold]
        fixs = [b for b in self.banks if b.balance > 0 and b.balance > self.irs_threshold]

        floats.sort()
        fixs.sort(reverse=True)

        for (fx,fl) in itertools.izip(fixs,floats):
            self.create_swap(fl,fx)

class SortedRandomIRSMarket(Market):

    def create_swaps(self):
        floats = [b for b in self.banks if b.balance < 0 and fabs(b.balance) > self.irs_threshold]
        fixs = [b for b in self.banks if b.balance > 0 and b.balance > self.irs_threshold]

        floats.sort()
        fixs.sort(reverse=True)

        for (fx,fl) in itertools.izip(fixs,floats):
            irs_val = min(self.max_irs_value,min(abs(fl.balance),fx.balance))

            if(self.generator.rand() < (irs_val/self.max_irs_value)):
                self.create_swap(fl,fx)

class ShuffleRandomIRSMarket(Market):

    def create_swaps(self):
        floats = [b for b in self.banks if b.balance < 0 and fabs(b.balance) > self.irs_threshold]
        fixs = [b for b in self.banks if b.balance > 0 and b.balance > self.irs_threshold]

        self.generator.shuffle(floats)
        self.generator.shuffle(fixs)

        for (fx,fl) in itertools.izip(fixs,floats):
            irs_val = min(self.max_irs_value,min(abs(fl.balance),fx.balance))

            if(self.generator.rand() < (irs_val/self.max_irs_value)):
                self.create_swap(fl,fx)
