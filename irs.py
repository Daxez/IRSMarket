"""
old
"""
"""IRS class: class representing an IRS"""
# Author: Dexter Drupsteen

from __future__ import division

import math
import logging
import numpy as np
import basegraphs as bg

class IRS(bg.Edge):
    def __init__(self,value,tenure,start_time):
        super(IRS,self).__init__()
        self.__value__ = value
        self.tenure = tenure
        self.time = 0
        self.start_time = start_time

    @property
    def market(self):
        return self.graph

    @property
    def float_value(self):
        # Gives back the correction value. + for floating because fixed is negative
        # a positive value will pull it back to zero
        return self.__value__

    @property
    def fix_value(self):
        # Gives back the correction value. - for fixed because fixed is positive
        # a negative value will pull it back to zero
        return -1.0*self.__value__

    @property
    def fixed(self):
        return self.left

    @property
    def floating(self):
        return self.right

    def activate(self,floating,a_id):
        if(floating):
            self.fixed.activate(a_id)
        else:
            self.floating.activate(a_id)

    def time_step(self):
        self.time += 1
        if(self.time == self.tenure):
            self.remove()

    def remove(self):
        self.market.datacontainer.register_swap(self.name,self.floating.name,
            self.fixed.name,self.__value__,self.start_time,
            self.start_time+self.time,self.tenure)

        self.floating.set_dirty()
        self.fixed.set_dirty()

        super(IRS,self).remove()
