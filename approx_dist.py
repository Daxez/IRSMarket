"""
file containing methods used to estimate distributions.
"""

from __future__ import division

import sys
import warnings
import time
import pickle
import numpy as np
import math
import scipy.optimize
import matplotlib.pyplot as pplot

from matplotlib.mlab import normpdf
from scipy.special import zeta
from scipy.stats import norm

from data.models import *

def pl(alpha,x):
    return ((1/zeta(alpha,1))*x**(-alpha))

def distribution(x,w,alpha,mu,sigma):
    return w*((1/zeta(alpha,1))*x**(-alpha)) + (1-w)*normpdf(x,mu,sigma)

def diff_distribution(x,w,alpha,mu,sigma):
    return (((w-1) * (x-mu) * np.exp(-(x-mu)**2/(2*sigma**2)))/(np.sqrt(2*np.pi)*(sigma**2)**(3/2)))-(alpha*w*x**(-alpha-1))/(zeta(alpha,1))

def get_normal_dist(x,y):
    mu = sum(x*y)
    sigma = np.sqrt(sum((x-mu)**2 * y))

    return mu, sigma

def to_minimize(p,x,y):
    w = p[0]
    alpha = p[1]
    mu = p[2]
    sigma = p[3]

    e = sum(abs(distribution(x,w,alpha,mu,sigma)-y))
    return e

def get_aggregate_dist(aggregate_id):

    session = get_session()
    run = session.query(RunModel).filter(RunModel.aggregate_id == aggregate_id).first()

    if(run == None):
        print "Could not find a run with aggregate_Id %s, now exiting"%aggregate_id
        exit(0)

    apl = session.query(AggregatePowerlaw)\
           .filter(AggregatePowerlaw.aggregate_id == aggregate_id)\
           .first()

    freqs = session.execute("""SELECT size, frequency
                               FROM default_aggregate
                               WHERE aggregate_id = :aggregate_id
                               ORDER BY size""",
                                  {'aggregate_id':aggregate_id}).fetchall()

    session.close()

    tot_s = sum([d[1] for d in freqs])
    x = np.array([d[0] for d in freqs])
    y = np.array([d[1] / tot_s for d in freqs])

    alpha = None
    if(apl != None):
        alpha = apl.alpha
    else:
        print "Warning, aggregate %s has no APL"%aggregate_id

    return x, y, alpha, run.no_banks

def redo_dists(x,y,tol=10):

    dists = [x[i+1] - x[i] for i in range(len(x)-1)]
    md = max(dists)
    if(md > tol):
        print "Extrapolating gap"
        mdi = np.where(dists == md)[0][0]

        nx = np.concatenate([x[:mdi],range(x[mdi],x[mdi+1]),x[mdi+1:]])
        ny = np.concatenate([y[:mdi],[y[mdi]]*(x[mdi+1]-x[mdi]),x[mdi+1:]])
        return nx,ny
    return x,y

def phi(x,mu,sigma):
    #'Cumulative distribution function for the standard normal distribution'
    return (1.0 + math.erf((x-mu)/(sigma*math.sqrt(2.0)))) / 2.0

def minimize_with_gap_tolerance(aggregate_id,m='SLSQP',tolerance=10):
    x,y,alpha,no_banks = get_aggregate_dist(aggregate_id)

    # Check if we need to split up stuff
    dists = [x[i+1] - x[i] for i in range(len(x)-1)]
    md = max(dists)
    if(md > tolerance):
        print "Extrapolating gap"
        mdi = np.where(dists == md)[0][0]

        x1 = x[:mdi]
        y1 = y[:mdi]
        x2 = x[mdi+1:]
        y2 = y[mdi+1:]

        print "Minimizing powerlaw"

        def minim(p,xs,ys):
            return sum(abs(pl(p,xs)-ys))

        x0 = np.array([3.0])
        mres = scipy.optimize.basinhopping(minim,x0,T=0.5,stepsize=0.1,
            minimizer_kwargs={'method':m,'args':(x1,y1),'jac':False},niter=1000)
        alpha = mres.x[0]
        print alpha

        # Normal distribution part
        print "Calculating normal distribution"
        y2n = y2 / sum(y2)
        mu = sum(y2n*x2)
        sigma = math.sqrt(sum(y2n*((x2 - mu)**2)))

        print "Mu %f, sigma: %f"%(mu,sigma)

        def minim(sw,alphas,mus,sigmas,xs,ys):
            return sum(abs(distribution(xs,sw,alphas,mus,sigmas)-ys))

        #x0 = np.array([0.8])
        #mres = scipy.optimize.basinhopping(minim,x0,T=0.5,stepsize=0.1,
        #    minimizer_kwargs={'args':(alpha,mu,sigma,x1,y1),'jac':False},niter=1000)
        w = 1-sum(y2)#mres.x[0]

    else:
        x0 = np.array([0.5,4,0.9*no_banks,0.1*no_banks])
        args = (x,y)

        bounds = ((0.0,1.0),(0.0,10),(0.0,no_banks),(0.0,None))
        mres = scipy.optimize.basinhopping(to_minimize,x0,T=2.0,stepsize=0.5,
            minimizer_kwargs={'method':m,'args':args,'bounds':bounds,'jac':False},niter=2000)


        w = mres.x[0]
        alpha = mres.x[1]
        mu = mres.x[2]
        sigma = mres.x[3]

    return w,alpha,mu,sigma, x, y

def minimize_and_get_turn_with_gap_tolerance(aggregate_id,m='SLSQP',tolerance=10):
    w,alpha,mu,sigma,x,y = minimize_with_gap_tolerance(aggregate_id,m,tolerance)

    x0 = None

    courseness = 0.5
    ys = diff_distribution(np.arange(1,max(x),courseness),w,alpha,mu,sigma)
    pys = np.where(ys > 0)[0]
    if(len(pys)>0):
        x0 = 1+(courseness*(pys[0]))

    if(x0 == None):
        print "Could not find root of derivative"

    return w,alpha,mu,sigma,x,y,x0

def minimize(aggregate_id, m='SLSQP'):

    x,y,alpha,no_banks = get_aggregate_dist(aggregate_id)

    args = redo_dists(x,y)

    x0 = np.array([0.5,alpha,0.9*no_banks,0.1*no_banks])

    #bounds = (slice(0.0,1.0,0.01),slice(0.0,10,0.01),(0.0,no_banks,0.1),(0.0,no_banks,0.1))
    #bounds = ((0.0,1.0),(0.0,10),(0.0,no_banks),(0.0,no_banks))
    bounds = ((0.0,1.0),(0.0,10),(0.0,no_banks),(0.0,None))

    minimized_res = scipy.optimize.basinhopping(to_minimize,x0,T=2.0,stepsize=0.5,
        minimizer_kwargs={'method':m,'args':args,'bounds':bounds,'jac':False},niter=2000)
    #minimized_res = scipy.optimize.basinhopping(to_minimize,minimized_res.x,T=1.0,stepsize=0.1,
    #    minimizer_kwargs={'method':m,'args':args,'bounds':bounds,'jac':False},niter=2000)
    #minimized_res = scipy.optimize.basinhopping(to_minimize,minimized_res.x,T=0.5,stepsize=0.01,
    #    minimizer_kwargs={'method':m,'args':args,'bounds':bounds,'jac':False},niter=2000)

    #minimized_res = scipy.optimize.differential_evolution(to_minimize,bounds, args=args)

    return minimized_res, x, y

def minimize_and_get_turn(aggregate_id,m=None):
    mres,x,y = minimize(aggregate_id,m)

    w = mres.x[0]
    alpha = mres.x[1]
    mu = mres.x[2]
    sigma = mres.x[3]

    x0 = None

    courseness = 0.5
    ys = diff_distribution(np.arange(1,max(x),courseness),w,alpha,mu,sigma)
    pys = np.where(ys > 0)[0]
    print pys
    if(len(pys)>0):
        x0 = 1+(courseness*(pys[0]))
    print x0

    if(x0 == None):
        print "Could not find root of derivative"

    return w,alpha,mu,sigma,x,y,x0

    try:
        x0 = scipy.optimize.newton(diff_distribution, 0.1*sigma, args=(w,alpha,mu,sigma), maxiter=1000)
        if(x0 > max(x)):
            raise Exception("Foolish thing")
    except:
        try:
            x0 = scipy.optimize.newton(diff_distribution, 10, args=(w,alpha,mu,sigma), maxiter=8000)
            if(x0 > max(x)):
                raise Exception("Foolish thing")
        except:
            print "Could not find root of derivative"

    return w,alpha,mu,sigma,x,y,x0

def minimize_all():
    session = get_session()
    runs = session.query(RunModel).all()
    session.close()

    aggregate_ids = list(set([r.aggregate_id for r in runs]))
    tot = len(aggregate_ids)
    res = []

    for (i,aggregate_id) in enumerate(aggregate_ids):
        print "Starting %d of %d"%((i+1),tot)
        start= time.time()

        mres,x,y = minimize(aggregate_id)

        w = mres.x[0]
        alpha = mres.x[1]
        mu = mres.x[2]
        sigma = mres.x[3]

        res.append(AggregateDistribution(aggregate_id,w,alpha,mu,sigma))

        fig = pplot.figure()
        ax = fig.add_subplot(111)

        ax.set_yscale('log')
        ax.set_xscale('log')

        ax.scatter(x,y)
        x_dist = np.linspace(1,max(x))
        ax.plot(x_dist,distribution(x_dist, w, alpha, mu, sigma,c),marker='o',c='r')

        ft = 'simulation_data/__optimize_aggregates_2/%s.png'%aggregate_id
        pplot.savefig(ft)
        pplot.close('all')

        print "Took %ds"%(time.time()-start)


    with open('simulation_data/approximations.bin','wb') as fp:
        pickle.dump(res,fp)

    session = get_session()
    session.bulk_save_objects(res)
    session.commit()
    session.close()

def minimize_and_calc_normal(aggregate_id):
    w,alpha,mu,sigma,x,y,x0 = minimize_and_get_turn_with_gap_tolerance(aggregate_id)

    print "w:%f, alpha: %f, mu: %f, sigma: %f"%(w,alpha,mu,sigma)

    ny = y[len(y)-len(np.where(x>=x0)[0]):]
    sny = sum(ny)
    ny = ny/(sum(ny))
    nx = x[np.where(x>=x0)[0]]

    m = None
    s = None
    wsum = None

    if(x0 != None):
        ratio_in_pl = 1-sny

        m,s= get_normal_dist(nx,ny)
        print x0
        wsum = sum(y[:x0])

    return w,alpha,mu,sigma,x,y,x0,m,s,wsum

def minimize_and_calc_normal_all():
    #session = get_session()
    #runs = session.query(RunModel).all()
    #session.close()

    #aggregate_ids = list(set([r.aggregate_id for r in runs]))
    aggregate_ids = ["b601e873-2e3e-40ea-b466-fb032b838699","3094a622-4d0c-400e-a4ec-52a52b11d3d2",
                     "095874a7-2dee-44ec-8036-4ee35e1bbaea","c53158f2-771c-48ba-ad0f-ce4305d41d93"]
    tot = len(aggregate_ids)
    res = []

    for (i,aggregate_id) in enumerate(aggregate_ids):
        print "Starting %d of %d"%((i+1),tot)
        start= time.time()

        w,alpha,mu,sigma,x,y,x0,m,s,wsum = minimize_and_calc_normal(aggregate_id)

        res.append(AggregateDistribution(aggregate_id,w,alpha,mu,sigma,x0,m,s,wsum))

        fig = pplot.figure()
        ax = fig.add_subplot(111)

        ax.set_yscale('log')
        ax.set_xscale('log')

        ax.scatter(x,y)
        x_dist = np.linspace(1,max(x))
        ax.plot(x_dist,distribution(x_dist, w, alpha, mu, sigma),marker='o',c='r')

        ft = 'simulation_data/__optimize_aggregates_2/%s.png'%aggregate_id
        pplot.savefig(ft)
        pplot.close('all')

        print "Took %ds"%(time.time()-start)


    with open('simulation_data/approximations2.bin','wb') as fp:
        pickle.dump(res,fp)

    session = get_session()
    session.bulk_save_objects(res)
    session.commit()
    session.close()

def show_minimize_diff(aggregate_id):

    w,alpha,mu,sigma,x,y,x0 = minimize_and_get_turn(aggregate_id)

    print "w:%f, alpha: %f, mu: %f, sigma: %f"%(w,alpha,mu,sigma)
    fig = pplot.figure()

    ax = fig.add_subplot(1,1,1)
    ax.set_yscale('log')
    ax.set_xscale('log')

    ax.scatter(x,y)
    x_dist = np.linspace(1,max(x))
    ax.plot(x_dist,distribution(x_dist, w, alpha, mu, sigma))
    ax.scatter([x0],distribution(np.array([x0]), w, alpha, mu, sigma),marker='^',c='r',s=120)

    pplot.show()

def show_normal_dist(aggregate_id):

    w,alpha,mu,sigma,x,y,x0 = minimize_and_get_turn_with_gap_tolerance(aggregate_id)

    if(x0 != None):
        print "w:%f, alpha: %f, mu: %f, sigma: %f, x0: %f"%(w,alpha,mu,sigma,x0)
    else:
        print "w:%f, alpha: %f, mu: %f, sigma: %f"%(w,alpha,mu,sigma)

    fig = pplot.figure()
    ax = fig.add_subplot(211)
    ax.set_xscale('log')
    ax.set_yscale('log')

    ny = y[len(y)-len(np.where(x>=x0)[0]):]
    sny = sum(ny)
    ny = ny/(sum(ny))
    nx = x[np.where(x>=x0)[0]]

    if(x0 != None):
        ratio_in_pl = 1-sny
        print "Ratio in powerlaw: %f, w: %f"%(ratio_in_pl,w)

        m,s = get_normal_dist(nx,ny)
        print "Mu: %f, Sigma: %f"%(m,s)
        ax.plot(nx,normpdf(nx,mu,sigma))

    if(len(nx) > 0):
        ax.scatter(nx,ny)

    ax = fig.add_subplot(212)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.scatter(x,y)
    x_dist = np.linspace(1,max(x),10000)
    ax.plot(x_dist,distribution(x_dist, w, alpha, mu, sigma))
    if x0 != None:
        ax.scatter([x0],distribution(np.array([x0]), w, alpha, mu, sigma),marker='^',c='r',s=120)

    ft = 'simulation_data/__optimize_aggregates_4/%s.png'%aggregate_id
    pplot.savefig(ft)
    pplot.close('all')

    return w,alpha,mu,sigma,x0

def show_multiple_minims():
    aggregate_ids = ['1ca2b336-c2bb-4336-aa28-1bcd3b292094',
                     '03e8de6d-11a0-4d32-8af3-831a6a716770',
                     '65caa671-27df-47f2-b59a-bfb5d252bc20',
                     '5d9be4ec-d8f1-41b9-b3e0-2fc082ed4266']

    methods = [None,
               'L-BFGS-B',
               'SLSQP',
               'TNC']

    fig = pplot.figure()
    cnt = 1

    for (j,meth) in enumerate(methods):
        for (i,aggregate_id) in enumerate(aggregate_ids):

            mres,x,y = minimize(aggregate_id,m=meth)

            w = mres.x[0]
            alpha = mres.x[1]
            mu = mres.x[2]
            sigma = mres.x[3]

            print "w:%f, alpha: %f, mu: %f, sigma: %f"%(w,alpha,mu,sigma)

            ax = fig.add_subplot(len(aggregate_ids)+1,len(methods),cnt)
            ax.set_yscale('log')
            ax.set_xscale('log')

            ax.scatter(x,y)
            x_dist = np.linspace(1,max(x))
            ax.plot(x_dist,distribution(x_dist, w, alpha, mu, sigma))
            cnt += 1

    pplot.show()

def show_minimize(aggregate_id,meth=None):
    fig = pplot.figure()

    mres,x,y = minimize(aggregate_id,m=meth)

    w = mres.x[0]
    alpha = mres.x[1]
    mu = mres.x[2]
    sigma = mres.x[3]

    print "w:%f, alpha: %f, mu: %f, sigma: %f"%(w,alpha,mu,sigma)

    ax = fig.add_subplot(111)
    ax.set_yscale('log')
    ax.set_xscale('log')

    ax.scatter(x,y)
    x_dist = np.linspace(1,max(x))
    ax.plot(x_dist,distribution(x_dist, w, alpha, mu, sigma))
    try:
        ax.plot(x_dist,diff_distribution(x_dist, w, alpha, mu, sigma))
    except:
        pass

    pplot.show()

def do_optimize_and_stuff():
    session = get_session()
    aggids = session.execute("""SELECT aggregate_id FROM aggregate_type WHERE type_id > 2""").fetchall()
    session.close()

    for (i,aggregate_id) in enumerate(aggids):
        print "%d of %d calcs"%(i+1,len(aggids))
        show_normal_dist(aggregate_id[0])

def optimize_for_start_aggs(firstletters):
    session = get_session()
    query = """SELECT DISTINCT aggregate_id from run
               WHERE aggregate_id not in (SELECT aggregate_id from aggregate_distribution) """
    aggids = session.execute(query).fetchall()
    session.close()

    for (i,aggid) in enumerate(aggids):
        print "Starting %d of %d"%(i+1, len(aggids))
        aggid = aggid[0]

        w,alpha,mu,sigma,x,y,x0 = minimize_and_get_turn_with_gap_tolerance(aggid)

        session = get_session()
        session.bulk_save_objects([AggregateDistribution(aggid, w, alpha, mu, sigma, x0, 0, 0, 0)])
        session.commit()
        session.close()

if __name__ == '__main__':
    warnings.simplefilter("ignore", Warning)

    #optimize_for_start_aggs([0,1,2,3])
    #optimize_for_start_aggs([4,5,6,7])
    #optimize_for_start_aggs([8,9,'a','b'])
    optimize_for_start_aggs(['e','f','g','c','d'])


    exit()
    do_optimize_and_stuff()
    exit()

    if(len(sys.argv) > 1):
        aggregate_id = sys.argv[1]
    else:
        aggregate_id = '03e8de6d-11a0-4d32-8af3-831a6a716770'

    show_normal_dist(aggregate_id)
