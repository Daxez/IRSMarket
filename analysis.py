"""
Module for doing analysis on the data 
It's pretty old, but not deleting it for possible usage elsewhere.
"""
from __future__ import division

import logging
import os
import math
import matplotlib.pyplot as pplot

from collections import defaultdict
from utils import Progress

# Balances through time
# Absolute net/gross through time
# Default size histogram
# Connectedness through time
# Waiting time distribution
# Average event per thousand

class Analysis:

    def __init__(self,root_path,save_to_file=True,prefix=""):
        if save_to_file and (not os.path.exists(root_path)):
            os.makedirs(root_path)

        self.root_path = root_path
        self.prefix = prefix
        self.save_to_file = save_to_file

    def balance_through_time(self,data):
        if(len(data.gross) <= 0 or len(data.net) <= 0):
            logging.info("No Balances, skipping swap plot")
            return

        logging.info("Generating balances through time")
        pgs = Progress(len(data.net))

        gross = defaultdict(float)
        net = defaultdict(float)

        for j,b in enumerate(data.net):
            for i,x in enumerate(data.net[b]):
                net[i] += math.fabs(x)

            for i,x in enumerate(data.gross[b]):
                gross[i] += math.fabs(x)

            pgs.update(j+1)

        for i in range(len(gross)):
            gross[i] = gross[i]/len(data.net)
            net[i] = net[i]/len(data.net)

        pplot.figure()
        pplot.plot(gross.keys(),gross.values(),label="Gross")
        pplot.plot(net.keys(),net.values(),label="net")
        pplot.title("Absolute net and gross through time")

        if(self.save_to_file):
            if(not os.path.exists(self.root_path)):
                os.makedirs(self.root_path)

            pplot.savefig(self.root_path+('/%s_balance_tt.png'%self.prefix))
        else:
            pplot.show()
        pplot.close()

        print ""


    def no_swaps_through_time(self,data):
        if len(data.swaps) <= 0:
            logging.info("No swaps, skipping swap plot")
            return

        dividedby = 1
        if(len(data.banks) > 0):
            dividedby = len(data.banks)

        logging.info("Generating swaps through time")
        maxval = len(data.swaps)
        pgs = Progress(maxval)
        rng = [0]*data.steps

        for (j,(x,y,z,start,end,ten)) in enumerate(data.swaps.values()):
            for i in range(end-start):
                rng[i+start] += 1/dividedby
            pgs.update(j+1)

        avgswaps = sum(rng)/data.steps

        pplot.figure()
        pplot.plot(range(len(rng)),rng)
        pplot.title("Swaps per bank through time, average through time: %f"%avgswaps)

        if(self.save_to_file):
            if(not os.path.exists(self.root_path)):
                os.makedirs(self.root_path)

            pplot.savefig(self.root_path+('/%s_swaps_tt.png'%self.prefix))
        else:
            pplot.show()
        pplot.close()

        print ""

    def irs_tries_distribution(self,data):
        if len(data.irs_tries) <= 0:
            logging.info("No irs tries, skipping distribution")

        logging.info("Generating irs tries histogram")

        res = [0]*len(data.irs_tries)
        s = 0
        l = 0
        for (i,k) in enumerate(data.irs_tries):
            res[i] = len(data.irs_tries[k])/sum(data.irs_tries[k])
            s += sum(data.irs_tries[k])
            l += len(data.irs_tries[k])

        pplot.figure()
        pplot.hist(res)

        tot_avg = 0
        if(s != 0):
            tot_avg = l/s
        pplot.title("IRS creation chance. Average: %f"%tot_avg)

        if(self.save_to_file):
            pplot.savefig(self.root_path+('/%s_irs_try_hist.png'%self.prefix))
        else:
            pplot.show()
        pplot.close()



    def default_histogram(self,data):
        if len(data.defaults) <= 0:
            logging.info("No Defaults, skipping Default plot")
            return

        logging.info("Generating default histogram")
        s_lst = [d[2] for d in data.defaults.values()]

        pplot.figure()
        pplot.hist(s_lst)
        pplot.title("Default avalanche sizes")

        if(self.save_to_file):
            pplot.savefig(self.root_path+('/%s_default_hist.png'%self.prefix))
        else:
            pplot.show()
        pplot.close()

        ##### Continue with distribution for 1.

        length = len(s_lst)
        vals = defaultdict(float)
        for s in s_lst:
            vals[s] += 1/length

        largest_size = max(vals.keys())

        fig = pplot.figure()
        ax = fig.add_subplot(1,1,1)

        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.scatter(vals.keys(),vals.values())

        ax.set_xlim((0,largest_size))
        ax.set_ylim((0,max(vals.values())))
        ax.set_title("Avalanche Size Distribution")

        if(self.save_to_file):
            pplot.savefig(self.root_path+('/%s_default_distr_scatter.png'%self.prefix))
        else:
            pplot.show()
        pplot.close()


    def links_at_defaults(self,data):
        if(len(data.defaults) <= 0):
            logging.info("No defaults, skipping links at defaults")
            return

        conne = [x[3] for x in data.defaults.values()]

        pplot.figure()
        pplot.hist(conne,bins=30)
        pplot.title("Number of connections during defaults")

        if(self.save_to_file):
            pplot.savefig(self.root_path+('/%s_default_connects.png'%self.prefix))
        else:
            pplot.show()
        pplot.close()

    def tenure_and_value_histograms(self,data):
        if(len(data.swaps) <= 0):
            logging.info("No swaps, skipping swap histograms")

        swap_tenures = [x[5] for x in data.swaps.values()]
        swap_values = [x[2] for x in data.swaps.values()]

        f,(ax1,ax2) = pplot.subplots(2)

        ax1.set_title("Swap Tenure Times")
        ax1.hist(swap_tenures,bins=70)

        ax2.set_title("Swap Values")
        ax2.hist(swap_values,bins=35)

        if(self.save_to_file):
            pplot.savefig(self.root_path+('/%s_swap_tenure_and_value.png'%self.prefix))
        else:
            pplot.show()
        pplot.close()

    def most_defaulted_bank_info(self,data):
        if(len(data.defaults) <= 0):
            logging.info("No defaults, skipping MDB info")
            return

        logging.info("Going to generate most defaulted bank info")

        bankdict = defaultdict(int)
        for (bankid,time,no,swps) in data.defaults.values():
            bankdict[bankid] += 1

        most_defaulted = max(bankdict, key=lambda x: bankdict[x])

        # Balances
        net = data.net[most_defaulted]
        gross = data.gross[most_defaulted]
        x = range(len(net))

        # Swaps
        swaps = [0] * len(net)
        for (fl,fx,v,s,e,t) in data.swaps.values():
            if(fl == most_defaulted or fx == most_defaulted):
                l = range(e - s)
                for i in l:
                    if(s+i < len(swaps)):
                        swaps[s+i] += 1


        f,(ax1,ax2) = pplot.subplots(2)
        ax1.set_title("Net and Gross balances")
        ax1.plot(x,net,label="Net")
        ax1.plot(x,gross,label="Gross")
        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(handles, labels,loc=2)

        ax2.set_title("Number of swaps through time")
        ax2.plot(x,swaps)

        if(self.save_to_file):
            pplot.savefig(self.root_path+('/%s_MDB_Info.png'%self.prefix))
        else:
            pplot.show()
        pplot.close()

    def number_of_defaults(self,data):
        if(len(data.defaults) <= 0):
            logging.info("No defaults, skipping Number of defaults")

        logging.info("Going to generate number of defaults")

        nd = [0]*data.steps
        rng = range(data.steps)
        for (bid,t,n,ns) in data.defaults.values():
            nd[t] += 1

        pplot.figure()
        pplot.plot(rng,nd)
        pplot.title("Number of defaults per timestep")

        if(self.save_to_file):
            pplot.savefig(self.root_path+('/%s_no_defaults_tt.png'%self.prefix))
        else:
            pplot.show()
        pplot.close()

    def average_degree(self,data):
        if(len(data.degree) <= 0):
            logging.info("No degree, skipping average degree")

        #Per bank, per timestep degree
        degrees = data.degree

        # Average through time
        nsteps = len(degrees[degrees.keys()[0]])
        nb = len(degrees.keys())
        avg = [0]*nsteps

        rng = range(nsteps)

        for i in rng:
            for b in degrees.keys():
                avg[i] += (degrees[b][i]/nb)

        all_t_avg = sum(avg)/nsteps

        pplot.figure()
        pplot.plot(rng,avg)
        pplot.title("Average degree through time (overall: %f)"%all_t_avg)

        if(self.save_to_file):
            pplot.savefig(self.root_path+('/%s_average_degree_tt.png'%self.prefix))
        else:
            pplot.show()
        pplot.close()

        print "Calculating average degree per bank"

        average_degree_per_bank = [0]*nb
        for (i,b) in enumerate(degrees.keys()):
            average_degree_per_bank[i] = sum(degrees[b])/nsteps

        pplot.figure()
        pplot.hist(average_degree_per_bank)
        pplot.title("Average degree per bank")

        if(self.save_to_file):
            pplot.savefig(self.root_path+('/%s_average_degree_pb.png'%self.prefix))
        else:
            pplot.show()
        pplot.close()
