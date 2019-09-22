"""
implementation of the simulation specifically for PYPY.
I recommend running using quick_sweep though (that's a bit easier to configure)
this file is way bigger than it needs to be because of all the stuff we are collecting (or might be collecting)
The core of the simulation should be no bigger than 200 lines.
"""
from __future__ import division

import time
import itertools
import os
import pickle
from collections import defaultdict, Counter, deque
from uuid import uuid4
from math import floor, log
import json

import numpy as np
from utils import Progress, is_valid_uuid
from datacontainer import DataContainer
from data.models import *

from avalanche_progression import AvalancheProgression
from critical_utilities import *

class irs:
    def __init__(self,value,start,tenure,a,b):
        self.name = str(uuid4())
        self.value = value
        self.start = start
        self.end = start+tenure
        self.a = a
        self.b = b

class sim:
    def __enter__(self):
        return self

    def __exit__(self,esc_type,exc_value,traceback):
        del self.balances
        del self.irss
        del self.positive_irss
        del self.negative_irss

    def __init__(self,config,dc,update_function,save_risk,save_dist, scatter_moments = []):
        self.dc = dc
        self.no_banks = config['no_banks']
        self.no_steps = config['no_steps']
        self.step_update = 10**floor(log(0.01*self.no_steps,10))
        self.T = config['threshold']
        self.s = config['sigma']
        self.v = config['max_irs_value']
        self.ten = config['max_tenure']
        self.dissipation = config['dissipation']
        self.min_progression_size = 0.5*self.no_banks

        self.balances = np.zeros(self.no_banks,dtype=float)
        self.gross_balances = np.zeros(self.no_banks,dtype=float)
        self.irss = {}
        self.positive_irss = defaultdict(dict)
        self.negative_irss = defaultdict(dict)
        self.pgs_update = update_function
        self.scatter_moments = scatter_moments
        self.save_avalanche_progression = False
        self.avalanche_progressions = []
        self.save_giant_component = False
        self.save_degree_distribution = False
        self.seed = None
        self.save_degree_on_default = False
        self.degrees_on_default = []
        self.save_default_rate = False
        self.last_default_time = [0 for i in xrange(self.no_banks)]
        self.default_rate = []
        self.time = 0
        self.save_time_between_large_events = False
        self.tble = []
        self.last_large_event = 0

        self.connection_scatters = []

        self.save_dist = save_dist
        if(save_dist):
            self.trials = []

        self.save_risk = save_risk
        self.save_risk_avalanche_time_series = False
        self.max_default_size_t = np.zeros(self.no_steps)
        if(save_risk):
            self.risk = np.zeros(self.no_steps)

        self.critical_info = []
        self.collect_critical_info = False

    def create_irs(self,start,a,b):
        i = irs(self.v,start,self.ten,a,b)
        self.irss[i.name] = i
        self.positive_irss[a][i.name] = i
        self.negative_irss[b][i.name] = i

        self.balances[a] += self.v
        self.balances[b] -= self.v

    def destroy_irs(self,name,default=False):
        i = self.irss[name]
        a = 1
        if default:
            a = 1-self.dissipation

        self.balances[i.a] -= a*i.value
        self.balances[i.b] += a*i.value

        del self.positive_irss[i.a][i.name]
        del self.negative_irss[i.b][i.name]
        del self.irss[name]
        del i

    def calc_giant_component_size(self,nodes):
        if len(nodes) == 0: return 0
        components = []

        for b in nodes:
            if len([c for c in components if b in c]) > 0:
                continue

            nodes_to_do = deque([b])
            neighbors = set([b])
            nodes_done = []
            while len(nodes_to_do) > 0:
                b = nodes_to_do.pop()
                nodes_done.append(b)

                affected = [i.b for i in self.positive_irss[b].values() if i.b in nodes]
                affected += [i.a for i in self.negative_irss[b].values() if i.a in nodes]

                af = set(affected)
                neighbors = neighbors.union(af)
                ntd = af.difference(nodes_done)
                nodes_to_do.extend(ntd)

            components.append(list(neighbors))

        return max([len(c) for c in components])

    def get_degree(self,bank,filter_by=[]):
        if(len(filter_by) > 0):
            affected = [i.b for i in self.positive_irss[bank].values() if i.b in filter_by]
            affected += [i.a for i in self.negative_irss[bank].values() if i.a in filter_by]
        else:
            affected = [i.b for i in self.positive_irss[bank].values()]
            affected += [i.a for i in self.negative_irss[bank].values()]

        return len(set(affected))

    def get_max_degrees(self, bank, filter_by=[]):
        pos_degs = None
        neg_degs = None

        if len(filter_by) == 0:
            pos_degs = Counter([x.b for x in self.positive_irss[bank].values()])
            neg_degs = Counter([x.a for x in self.negative_irss[bank].values()])
        else:
            pos_degs = Counter([x.b for x in self.positive_irss[bank].values() if x.b in filter_by])
            neg_degs = Counter([x.a for x in self.negative_irss[bank].values() if x.a in filter_by])

        max_pos_deg = 0
        max_neg_deg = 0

        if(len(pos_degs) > 0 ):
            max_pos_deg = pos_degs[max(pos_degs,key=pos_degs.get)]

        if(len(neg_degs) > 0 ):
            max_neg_deg = neg_degs[max(neg_degs,key=neg_degs.get)]

        return max_pos_deg,max_neg_deg

    def get_critical_nodes(self):
        critical_nodes = []
        all_banks = range(self.no_banks)
        for i in all_banks:
            max_pos_deg,max_neg_deg = self.get_max_degrees(i)
            pos_hedged = max_pos_deg*self.v
            neg_hedged = max_neg_deg*self.v

            if self.balances[i] + neg_hedged > self.T or self.balances[i] - pos_hedged < -self.T:
                critical_nodes.append(i)
        return critical_nodes


    def get_critical_banks(self,time):
        degrees = []
        critical_nodes = self.get_critical_nodes()

        non_critical_nodes = list(set(all_banks).difference(critical_nodes))

        critical_degrees = []
        ctnc_degrees = []

        super_critical_nodes = []

        for i in critical_nodes:
            critical_degrees.append(self.get_degree(i,critical_nodes))
            ctnc_degrees.append(self.get_degree(i,non_critical_nodes))

            max_pos_deg,max_neg_deg = self.get_max_degrees(i, critical_nodes)
            pos_hedged = max_pos_deg*self.v
            neg_hedged = max_neg_deg*self.v
            if self.balances[i] + neg_hedged > self.T or self.balances[i] - pos_hedged < -self.T:
                super_critical_nodes.append(i)

        nctc_degrees = []
        non_critical_degrees = []
        #for i in non_critical_nodes:
        #    nctc_degrees.append(self.get_degree(i,critical_nodes))
        #    non_critical_degrees.append(self.get_degree(i,non_critical_nodes))

        if(self.save_giant_component):
            self.giant_components[time] = self.calc_giant_component_size(critical_nodes)

        return CriticalInfo(len(critical_nodes),
                            len(super_critical_nodes),
                            critical_degrees,
                            ctnc_degrees,
                            nctc_degrees,
                            non_critical_degrees,
                            degrees)

    def default(self,bank):
        track_avalanche = self.current_avalanche_progression != None
        cnt = 0
        self.default_affected[bank] = 1
        defaulting = deque()
        defaulting.append((bank,0,None))
        defaulted = [bank]
        tree_info = {
            'nodes':[],
            'links':[]
        }

        #Administration
        if self.save_avalanche_tree or track_avalanche:
            critical_nodes = self.get_critical_nodes()
        if(track_avalanche):
            self.current_avalanche_progression.critical_nodes = critical_nodes

        while(len(defaulting) > 0):
            #Administration
            if(track_avalanche):
                dist = [self.balances[z] for z in xrange(len(self.balances)) if not z in defaulted]
                self.current_avalanche_progression.add_distribution(np.array(dist))

            # Get current defaulting node
            current_defaulting,level,src = defaulting.pop()

            # Put it in Avalanche tree info
            if self.save_avalanche_tree:
                tree_info['nodes'].append({
                    'node': current_defaulting,
                    'level': level,
                    'critical': current_defaulting in critical_nodes})
                tree_info['links'].append((src,current_defaulting))

            #Get affected nodes
            affected = [i.b for i in self.positive_irss[current_defaulting].values()]
            affected += [i.a for i in self.negative_irss[current_defaulting].values()]

            # Destroy all its irs's
            map(lambda x: self.destroy_irs(x.name,True),
                self.positive_irss[current_defaulting].values())
            map(lambda x: self.destroy_irs(x.name,True),
                self.negative_irss[current_defaulting].values())

            #Get all nodes affected which are not yet defaulted (but not processed)
            #A node can default and not be processed, so its links are still in existence
            unique_affected = [i for i in set(affected) if not i in defaulted]

            # Some administration
            no_defaulted_by_this_one = 0
            no_critical_defaulted_by_this_one = 0

            for i in unique_affected:
                self.default_affected[i] = 1
                b = self.balances[i]

                if (b < -self.T  or b > self.T):
                    #Put it in the queue
                    defaulting.appendleft((i, level+1,current_defaulting))
                    defaulted.append(i)

                    # Administration
                    if (track_avalanche):
                        no_defaulted_by_this_one += 1
                        if(i in critical_nodes):
                            no_critical_defaulted_by_this_one += 1
                elif(track_avalanche):
                    # Administration
                    self.current_avalanche_progression.no_affected[i] += 1

            #Administration
            if(track_avalanche):
                self.current_avalanche_progression.add_defaulted(no_defaulted_by_this_one,
                                                                 no_critical_defaulted_by_this_one,
                                                                 len(unique_affected))
                self.current_avalanche_progression.default_order = defaulted

            #Reset balances
            if (self.save_abs_risk_and_dissipation):
                self.balance_dissipation[self.time] += abs(self.gross_balances[current_defaulting])
            self.balances[current_defaulting] = 0
            self.gross_balances[current_defaulting] = 0


        #Administration
        if(self.save_avalanche_tree and len(defaulted) > 0.7*self.no_banks):
            filename = '%d_%s.json'%(len(defaulted),str(uuid4()))
            path = '%s%s'%(self.avalanche_tree_file_path,filename)
            print len(defaulted)
            with open(path,'w') as fp:
                json.dump(tree_info,fp,indent=4)

        if(self.save_default_rate):
            for d in defaulted:
                self.default_rate.append(self.time - self.last_default_time[d])
                self.last_default_time[d] = self.time

        return len(defaulted)

    def run(self):
        np.random.seed(self.seed)
        pbalance = 0
        potential_density = (self.no_banks*(self.no_banks-1))/2
        for tme in xrange(self.no_steps):
            self.time = tme

            for i in self.irss.values():
                if tme > i.end:
                    self.destroy_irs(i.name)

            p_total_gross_balance = np.sum(abs(self.gross_balances))
            epsilon = np.random.normal(0, self.s, self.no_banks)

            self.balances += epsilon
            self.gross_balances += epsilon

            total_gross_balance = np.sum(abs(self.gross_balances))
            if (self.save_abs_risk_and_dissipation):
                self.abs_added_risk[tme] = total_gross_balance - p_total_gross_balance

            will_be_a = []
            will_be_b = []
            for i in xrange(self.no_banks):
                b = self.balances[i]
                if(b < -self.v and b < 0):
                    will_be_a.append(i)
                elif(b > self.v):
                    will_be_b.append(i)

            np.random.shuffle(will_be_a)
            np.random.shuffle(will_be_b)

            map(lambda x: self.create_irs(tme, x[0],x[1]),zip(will_be_a,will_be_b))

            no_irs_per_bank = [0]*self.no_banks

            total_abs_balances_before_default = np.sum(abs(self.balances))
            density_before_defaults = (sum([self.get_degree(i) for i in xrange(self.no_banks)])/2)/potential_density
            gross_risk_before_default = float(sum(abs(self.gross_balances)))
            average_degree = sum([self.get_degree(i) for i in xrange(self.no_banks)])/self.no_banks

            for i in xrange(self.no_banks):
                b = self.balances[i]

                if(b < -self.T and b < 0) or b > self.T:
                    defaulting_degree = self.get_degree(i)

                    self.default_affected = np.zeros(self.no_banks)
                    self.current_avalanche_progression = AvalancheProgression()

                    ds = self.default(i)

                    self.dc.register_default_simple(ds)
                    if(ds >= 0.5*self.no_banks):
                        self.tble.append(self.time-self.last_large_event)
                        self.last_large_event = self.time

                    if(self.save_degree_on_default):
                        self.degrees_on_default.append((defaulting_degree,ds))

                    if(self.save_average_degree_on_default):
                        self.average_degree_on_default[ds].append(average_degree)
                        self.degree_on_default[ds].append(defaulting_degree)

                    if(self.save_avalanche_progression and ds > self.min_progression_size):
                        self.current_avalanche_progression.size = ds
                        self.avalanche_progressions.append(self.current_avalanche_progression)
                        self.current_avalanche_progression = None

                    if self.save_risk_avalanche_time_series:
                        self.max_default_size_t[tme] = max(ds,self.max_default_size_t[tme])
                    if self.save_dist:
                        da = int(sum(self.default_affected))
                        self.trials.append((ds,da))
                    if self.save_density_for_avalanche_size:
                        self.density_per_avalanche_size[ds].append(density_before_defaults)
                    if self.save_gross_risk_for_avalanche_size:
                        self.gross_risk_per_avalanche_size[ds].append(gross_risk_before_default)

                no_irs_per_bank[i] = len(self.positive_irss[i].values())
                no_irs_per_bank[i] += len(self.negative_irss[i].values())

            if(self.save_risk):
                self.risk[tme] = sum(abs(self.gross_balances))

            if(tme in self.scatter_moments):
                self.connection_scatters.append(self.get_connection_scatter())

            if(self.collect_critical_info):
                self.critical_info.append(self.get_critical_banks(tme))

            if(self.save_degree_distribution):
                self.save_degrees(tme,no_irs_per_bank)

            #if(self.save_abs_risk_and_dissipation):
                #self.balance_dissipation[self.time] = total_abs_balances_before_default - np.sum(abs(self.balances))
                #print "Dissipated",total_gross_balance - np.sum(abs(self.gross_balances))

            if tme % self.step_update == 0:
                self.pgs_update(tme)

    def save_degrees(self,tme,no_irs_per_bank):
        self.no_irs[tme,:] = no_irs_per_bank
        self.degrees[tme,:] = [self.get_degree(i) for i in xrange(self.no_banks)]

    def get_connection_scatter(self):
        pairs = []

        for k in self.irss:
            balance_1 = float(self.balances[self.irss[k].a])
            balance_2 = float(self.balances[self.irss[k].b])
            pairs.append((balance_1,balance_2))

        return pairs

if(__name__ == '__main__'):
    steps = 100000
    #steps = 300000
    save = False
    save_risk = False
    save_risk_avalanche_time_series = False
    save_dist = False
    save_giant_component = False
    save_avalanche_progression = False
    save_critical_info = False
    save_avalanche_tree = False
    save_degree_distribution = False
    save_degree_on_default = False
    save_default_rate = False
    save_time_between_large_events = False
    save_abs_risk_and_dissipation = False
    save_density_for_avalanche_size = False
    save_gross_risk_for_avalanche_size = False
    save_average_degree_on_default = True
    no_connection_scatter_moments = 0
    connection_scatter_moments = np.random.randint(0,steps,no_connection_scatter_moments)

    dcconfig = {
        'model':{
            'no_banks' : 100,
            'no_steps' : steps,
            'threshold' :  10,
            'sigma' : 1,
            'max_irs_value' : 7,#4,
            'irs_threshold' : -1,
            'dissipation' : 0.0,
            'max_tenure' : 400
        },
        'analysis':{
            'data_to_save':['defaults']
        },
        'file_root':'./simulation_data/',
        'market_type':7
    }

    dc = DataContainer(dcconfig,str(uuid4()),str(uuid4()))
    p = Progress(steps)

    s = sim(dcconfig['model'],dc,p.update,save_risk,save_dist,connection_scatter_moments)
    s.save_degree_distribution = save_degree_distribution
    if(s.save_degree_distribution):
        s.degrees = np.zeros((steps,dcconfig['model']['no_banks']))
        s.no_irs = np.zeros((steps,dcconfig['model']['no_banks']))
    s.save_avalanche_progression = save_avalanche_progression
    s.save_risk_avalanche_time_series = save_risk_avalanche_time_series
    s.collect_critical_info = save_critical_info
    s.save_giant_component = save_giant_component
    s.save_avalanche_tree = save_avalanche_tree
    s.avalanche_tree_file_path = './simulation_data/trees/%s/'%dc.aggregate_id
    s.save_degree_on_default = save_degree_on_default
    s.save_default_rate = save_default_rate
    s.save_time_between_large_events = save_time_between_large_events
    s.save_abs_risk_and_dissipation = save_abs_risk_and_dissipation
    s.save_average_degree_on_default = save_average_degree_on_default

    if(save_average_degree_on_default):
        s.average_degree_on_default = defaultdict(list)
        s.degree_on_default = defaultdict(list)

    if(save_abs_risk_and_dissipation):
        s.balance_dissipation = np.zeros(steps)
        s.abs_added_risk = np.zeros(steps)

    s.save_density_for_avalanche_size = save_density_for_avalanche_size
    if(s.save_density_for_avalanche_size):
        s.density_per_avalanche_size = defaultdict(list)

    s.save_gross_risk_for_avalanche_size = save_gross_risk_for_avalanche_size
    if(s.save_gross_risk_for_avalanche_size):
        s.gross_risk_per_avalanche_size = defaultdict(list)
        s.gross_risk_per_avalanche_size = defaultdict(list)
    

    if(s.save_avalanche_tree):
        os.makedirs(s.avalanche_tree_file_path)

    if(save_giant_component): s.giant_components = np.zeros(s.no_steps)

    start = time.time()

    p.start()
    s.run()
    p.finish()

    print
    print "Run took %d seconds"%(time.time()-start)

    if(save):
        print "Saving data"
        dc.save_defaults()
        dc.save_run()

    if s.save_avalanche_progression:
        print "Saving avalanche progression"
        file_path = './simulation_data/avalanche_progression/%s.bin'%dc.aggregate_id
        with file(file_path,'wb') as fp:
            pickle.dump(s.avalanche_progressions,fp)
            pickle.dump(dcconfig,fp)

    if s.collect_critical_info:
        print "Critical info"
        file_path = './simulation_data/critical/%s.bin'%dc.aggregate_id
        with file(file_path,'wb') as fp:
            pickle.dump(s.critical_info,fp)
            pickle.dump(s.max_default_size_t.tolist(),fp)
            if(s.save_giant_component):
                pickle.dump(s.giant_components.tolist(),fp)
            pickle.dump(dcconfig,fp)

    if len(connection_scatter_moments) > 0:
        print "Connection Scatters"
        file_path = './simulation_data/connection_scatters/%s.bin'%dc.aggregate_id
        with file(file_path,'wb') as fp:
            pickle.dump(s.connection_scatters,fp)

    if save_dist:
        file_path = './simulation_data/dists/%s.bin'%dc.aggregate_id
        with file(file_path,'wb') as fp:
            pickle.dump(s.trials,fp)
            pickle.dump(dcconfig['model']['no_banks'],fp)

    if save_degree_distribution:
        file_path = './simulation_data/deg_distribution/%s.bin'%dc.aggregate_id
        with file(file_path,'wb') as fp:
            pickle.dump(s.degrees.tolist(),fp)
            pickle.dump(s.no_irs.tolist(),fp)
        file_path = './simulation_data/deg_distribution/%s.config'%dc.aggregate_id
        with file(file_path,'wb') as fp:
            json.dump(dcconfig, fp)

    if save_degree_on_default:
        file_path = './simulation_data/deg_default/%s.bin'%dc.aggregate_id
        with file(file_path,'wb') as fp:
            pickle.dump(s.degrees_on_default,fp)

    if save_default_rate:
        file_path = './simulation_data/default_rate/%s.bin'%dc.aggregate_id
        with file(file_path,'wb') as fp:
            pickle.dump(s.default_rate,fp)

    if save_risk_avalanche_time_series:
        file_path = './simulation_data/time_series/%s.bin'%dc.aggregate_id
        with file(file_path,'wb') as fp:
            pickle.dump(s.max_default_size_t.tolist(),fp)

    if save_time_between_large_events:
        file_path = './simulation_data/large_default_rate/%s.bin'%dc.aggregate_id
        with file(file_path,'wb') as fp:
            pickle.dump(dcconfig['model']['no_banks'],fp)
            pickle.dump(s.tble,fp)

    if save_abs_risk_and_dissipation:
        file_path = './simulation_data/risk_and_balance_dissipation/%s.bin'%dc.aggregate_id
        with file(file_path,'wb') as fp:
            pickle.dump(dcconfig,fp)
            pickle.dump(s.abs_added_risk.tolist(),fp)
            pickle.dump(s.balance_dissipation.tolist(),fp)

    if save_risk:
        file_path = './simulation_data/risk/%s.bin'%dc.aggregate_id
        with file(file_path, 'wb') as fp:
            pickle.dump(s.risk.tolist(), fp)
    
    if save_density_for_avalanche_size:
        file_path = './simulation_data/density_for_avalanche_size'
        abs_file_path = "%s/%s.bin"%(file_path, dc.aggregate_id)
        if(not os.path.exists(file_path)):
            os.makedirs(file_path)

        with file(abs_file_path, 'wb') as fp:
            pickle.dump(dcconfig['model'],fp)
            pickle.dump(s.density_per_avalanche_size,fp)

    if save_gross_risk_for_avalanche_size:
        file_path = './simulation_data/gross_risk_for_avalanche_size'
        abs_file_path = "%s/%s.bin"%(file_path, dc.aggregate_id)
        if(not os.path.exists(file_path)):
            os.makedirs(file_path)

        with file(abs_file_path, 'wb') as fp:
            pickle.dump(dcconfig['model'],fp)
            pickle.dump(s.gross_risk_per_avalanche_size,fp)


    print dc.aggregate_id
