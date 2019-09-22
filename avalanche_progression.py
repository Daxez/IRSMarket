"""
Old file for tracking avalanches
"""
from __future__ import division

from collections import defaultdict

class AvalancheProgression:

    def __init__(self):
        self.size = 0
        self.distributions = {}
        self.t = 0
        self.caused_defaults = []
        self.no_affected = defaultdict(int)
        self.critical_nodes = []
        self.default_order = []

    def add_distribution(self, distr):
        self.distributions[self.t] = distr.tolist()
        self.t += 1

    def add_defaulted(self,n,c,r):
        self.caused_defaults.append((n,c,r))
