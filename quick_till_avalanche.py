"""
Simulation variant that stops at the first large avalance
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

    def __init__(self,config,dc,update_function,save_risk,save_dist,
        scatter_moments = [], seed=None, start_measure = None,
        defaulting_bank_no = None, avalanche_fraction=0.9):

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
        self.seed = seed
        self.start_measure = start_measure
        self.defaulting_bank_no = defaulting_bank_no
        self.measuring = False
        self.avalanche_fraction = avalanche_fraction
        self.min_avalance_size = self.no_banks * self.avalanche_fraction

        self.defaulting_bank_balance = []

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

        return max(components, key=len)

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
                    defaulting.append((i, level+1,current_defaulting))
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
            self.balances[current_defaulting] = 0
            self.gross_balances[current_defaulting] = 0

        #Administration
        if(self.save_avalanche_tree and len(defaulted) > 0.5*self.no_banks):
            filename = '%d_%s.json'%(len(defaulted),str(uuid4()))
            path = '%s%s'%(self.avalanche_tree_file_path,filename)
            print len(defaulted)
            with open(path,'w') as fp:
                json.dump(tree_info,fp,indent=4)

        if(self.measuring):
            self.defaulted_nodes.append(defaulted)
        return len(defaulted)

    def run(self):
        quit = False
        avalanche_size = 0
        no_irs_per_bank = [0]*self.no_banks

        np.random.seed(self.seed)
        for tme in xrange(self.no_steps):
            if(self.start_measure != None and tme >= self.start_measure):
                self.measuring = True

            irs_removals = 0
            for i in self.irss.values():
                if tme > i.end:
                    self.destroy_irs(i.name)
                    irs_removals += 1
            epsilon = np.random.normal(0, self.s, self.no_banks)
            self.balances += epsilon
            self.gross_balances += epsilon

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

            matches = zip(will_be_a,will_be_b)
            map(lambda x: self.create_irs(tme, x[0],x[1]),matches)

            if(self.measuring):
                ctime = tme - self.start_measure
                self.save_degrees(ctime,no_irs_per_bank)
                self.measured_balances[ctime,:] = self.balances
                self.measured_gross_balances[ctime,:] = self.gross_balances
                #gc = self.calc_giant_component_size(range(self.no_banks))
                #self.giant_component.append(gc)
                self.network = self.get_network()
                self.irs_creations[ctime] = len(matches)
                self.irs_removals[ctime] = irs_removals
                self.irs_pb.append([(len(self.positive_irss[i].values()),
                        len(self.negative_irss[i].values())) for i in xrange(self.no_banks)])

            no_irs_per_bank = [0]*self.no_banks
            for i in xrange(self.no_banks):
                b = self.balances[i]

                if(b < -self.T and b < 0) or b > self.T:
                    self.default_affected = np.zeros(self.no_banks)
                    self.current_avalanche_progression = AvalancheProgression()
                    ds = self.default(i)
                    self.dc.register_default_simple(ds)

                    if(ds > self.min_avalance_size):
                        self.defaulting_bank_no = i
                        quit = True
                        avalanche_size = ds

                    if(self.save_avalanche_progression and ds > self.min_progression_size):
                        self.current_avalanche_progression.size = ds
                        self.avalanche_progressions.append(self.current_avalanche_progression)
                        self.current_avalanche_progression = None

                    if self.save_risk_avalanche_time_series:
                        self.max_default_size_t[tme] = max(ds,self.max_default_size_t[tme])
                    if self.save_dist:
                        da = int(sum(self.default_affected))
                        self.trials.append((ds,da))

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

            if tme % self.step_update == 0:
                self.pgs_update(tme)

            if(quit):
                break
        return tme, avalanche_size

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

    def get_network(self):
        A = np.zeros((self.no_banks,self.no_banks))

        for k in self.irss:
            irs = self.irss[k]
            A[irs.a, irs.b] += 1

        return A

def do_run(steps, no_banks, threshold, max_tenure, max_irs_value, avalanche_fraction=0.9):
    #steps = 10000
    save = False
    save_risk = False
    save_risk_avalanche_time_series = False
    save_dist = False
    save_giant_component = False
    save_avalanche_progression = False
    save_critical_info = False
    save_avalanche_tree = False
    save_degree_distribution = False
    no_connection_scatter_moments = 0
    connection_scatter_moments = np.random.randint(0,steps,no_connection_scatter_moments)

    seed = np.random.randint(0,1000)
    dcconfig = {
        'model':{
            'no_banks' : no_banks,
            'no_steps' : steps,
            'threshold' : threshold,
            'sigma' : 1,
            'max_irs_value' : max_irs_value,
            'irs_threshold' : -1,
            'dissipation' : 0.0,
            'max_tenure' : max_tenure
        },
        'analysis':{
            'data_to_save':['defaults']
        },
        'file_root':'./simulation_data/',
        'market_type':7,
        'seed':seed
    }

    measure_no_steps = 2*dcconfig['model']['max_tenure']

    ###########################################################################
    dc = DataContainer(dcconfig,str(uuid4()),str(uuid4()))
    p = Progress(steps)

    s = sim(dcconfig['model'],dc,p.update,save_risk,save_dist,
            connection_scatter_moments,seed, avalanche_fraction=avalanche_fraction)
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

    s.irs_creations = np.zeros(steps)
    s.irs_removals = np.zeros(steps)

    if(s.save_avalanche_tree):
        os.makedirs(s.avalanche_tree_file_path)

    if(save_giant_component): s.giant_components = np.zeros(s.no_steps)
    ###########################################################################


    start = time.time()
    p.start()
    tme, size = s.run()
    print
    p.finish()


    defaulting_bank = s.defaulting_bank_no
    start_at = tme - measure_no_steps + 1

    print "Large enough avalanche found at %d of size %d"%(tme,size)

    print
    print "Run took %d seconds"%(time.time()-start)
    print
    print "Going for the analysis"


    ###########################################################################
    ## Actual stuff thats needed
    dc = DataContainer(dcconfig,str(uuid4()),str(uuid4()))
    p = Progress(steps)

    s = sim(dcconfig['model'], dc, p.update, save_risk, save_dist,
            connection_scatter_moments,seed, start_at, defaulting_bank,
            avalanche_fraction=avalanche_fraction)

    nb = dcconfig['model']['no_banks']
    s.measured_balances = np.zeros((measure_no_steps,nb))
    s.measured_gross_balances = np.zeros((measure_no_steps,nb))
    s.degrees = np.zeros((measure_no_steps,nb))
    s.no_irs = np.zeros((measure_no_steps,nb))
    #s.giant_component = []
    s.defaulted_nodes = []
    s.irs_pb = []
    s.network = np.zeros((nb,nb))
    s.irs_creations = np.zeros(steps)
    s.irs_removals = np.zeros(steps)

    #################
    s.save_degree_distribution = save_degree_distribution
    s.save_avalanche_progression = save_avalanche_progression
    s.save_risk_avalanche_time_series = save_risk_avalanche_time_series
    s.collect_critical_info = save_critical_info
    s.save_giant_component = save_giant_component
    s.save_avalanche_tree = save_avalanche_tree
    s.avalanche_tree_file_path = './simulation_data/trees/%s/'%dc.aggregate_id
    if(s.save_avalanche_tree):
        os.makedirs(s.avalanche_tree_file_path)
    if(save_giant_component): s.giant_components = np.zeros(s.no_steps)
    ###########################################################################


    start = time.time()
    p.start()
    tme, size = s.run()
    p.finish()
    print
    print "Large enough avalanche found at %d of size %d"%(tme,size)

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

    if(True):
        os.makedirs("./simulation_data/large_avalanche_data/%s"%dc.aggregate_id)
        print "Saving stuff"
        file_path = './simulation_data/large_avalanche_data/%s/degrees.bin'%dc.aggregate_id
        with file(file_path,'wb') as fp:
            pickle.dump(s.degrees.tolist(),fp)

        file_path = './simulation_data/large_avalanche_data/%s/no_irs.bin'%dc.aggregate_id
        with file(file_path,'wb') as fp:
            pickle.dump(s.no_irs.tolist(),fp)
            pickle.dump(s.irs_pb,fp)

        file_path = './simulation_data/large_avalanche_data/%s/balances.bin'%dc.aggregate_id
        with file(file_path,'wb') as fp:
            pickle.dump(s.measured_balances.tolist(),fp)
            pickle.dump(s.measured_gross_balances.tolist(),fp)

        #file_path = './simulation_data/large_avalanche_data/%s/gc.bin'%dc.aggregate_id
        #with file(file_path,'wb') as fp:
        #    pickle.dump(s.giant_component,fp)

        file_path = './simulation_data/large_avalanche_data/%s/network.bin'%dc.aggregate_id
        with file(file_path,'wb') as fp:
            pickle.dump(s.network.tolist(),fp)

        file_path = './simulation_data/large_avalanche_data/%s/defaulted.bin'%dc.aggregate_id
        with file(file_path,'wb') as fp:
            pickle.dump(s.defaulted_nodes,fp)

        file_path = './simulation_data/large_avalanche_data/%s/irs_data.bin'%dc.aggregate_id
        with file(file_path,'wb') as fp:
            pickle.dump(s.irs_creations.tolist(),fp)
            pickle.dump(s.irs_removals.tolist(),fp)

        dcconfig['failed_bank'] = s.defaulting_bank_no
        file_path = './simulation_data/large_avalanche_data/%s/config.json'%dc.aggregate_id
        with open(file_path,'w') as fp:
            json.dump(dcconfig,fp,indent=4)

    print dc.aggregate_id

if __name__ == '__main__':
    steps = 100000
    #no_banks = 200
    no_banks = 100
    threshold = 15

    #max_irs_value = 6
    #max_irs_value = 3
    max_irs_value = 7
    #max_tenure = 250
    max_tenure = 400
    #avalanche_fraction = 0.8
    #avalanche_fraction = 0.9
    avalanche_fraction = 0.8


    do_run(steps, no_banks, threshold, max_tenure, max_irs_value,avalanche_fraction)
