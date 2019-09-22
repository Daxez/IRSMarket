"""

This used to be the Main application


"""
from __future__ import division
from market import Market,RandomIRSMarket
from math import fabs
from utils import Progress

import matplotlib.pyplot as pplot
import logging

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    steps = 50000
    step_rng = range(steps)
    number_of_agents = 50

    pgrs = Progress(steps)

    logging.info("Starting market simulation")

    mkt = RandomIRSMarket(number_of_agents,steps,False,pgrs.update)
    mkt.run()
    print ""

    logging.info("Ended market simulation, starting plotting")

    ## Plotting
    fig = pplot.figure()
    sx = 4
    sy = 3
    axn = 1
    subdim = (sy,sx)

    ax = pplot.subplot2grid(subdim,(0,0),colspan=2)
    ax.set_title("Bank balances")
    banks = mkt.nodes()
    for bank in banks:
        ax.plot(step_rng,bank.history)

    ax.set_xlim(0,steps)


    ax = pplot.subplot2grid(subdim,(0,2),colspan=2)
    ax.set_title("Number of unique neighbours")
    for b in banks:
        ax.plot(step_rng,b.unique_neighbours)

    ax.set_xlim(0,steps)


    ax = pplot.subplot2grid(subdim,(1,0),colspan=2)
    ax.set_title("Number of swaps in the system")
    number_of_swaps = [s[0] for s in mkt.irs_history]
    ax.plot(step_rng,number_of_swaps)
    ax.set_xlim(0,steps)


    ax = pplot.subplot2grid(subdim,(1,2),colspan=1)
    ax.set_title("Avalance size")
    ax.hist(mkt.avalanches.values(),label="S")


    ax = pplot.subplot2grid(subdim,(1,3),colspan=1)
    ax.set_title("Activation reach")
    ax.hist(mkt.activation_avalanches.values(),alpha=0.5,label="A")


    ax = pplot.subplot2grid(subdim,(2,0),colspan=2)
    ax.set_title("Number of links at default in time")
    for b in banks:
        x = []
        y = []
        for df in b.default_history:
            x.append(df[0])
            y.append(df[1])
        ax.scatter(x,y)
    ax.set_xlim(0,steps)


    ax = pplot.subplot2grid(subdim,(2,2),colspan=2)
    ax.set_title("Absolute gross and net balance")
    actual_balances = [0]*steps
    hedged_balances = [0]*steps

    for b in banks:
        for i in step_rng:
            actual_balances[i] += fabs(b.balance_history[i])
            hedged_balances[i] += fabs(b.history[i])

    actual_balances = [b/number_of_agents for b in actual_balances]
    hedged_balances = [b/number_of_agents for b in hedged_balances]

    hand1, = ax.plot(step_rng,actual_balances,label="Balance")
    hand2, = ax.plot(step_rng,hedged_balances,label="Hedged")
    ax.legend(handles=[hand1,hand2],loc=2)
    ax.set_xlim(0,steps)


    #fig.show()
    pplot.show()
    ## End of figure

    #raw_input("End of program, hit any key to exit")
