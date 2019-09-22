from __future__ import division
import os
import pickle
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as pplot
import matplotlib.cm as cm

def calc_for_t(t,s=50000):
    steps_array = []
    while(len(steps_array) < 100):
        es = np.random.normal(size=s)

        steps = 0
        sval = 0

        for e in es:
            sval += e
            steps += 1

            if abs(sval) > t:
                steps_array.append(steps)
                steps = 0
                sval = 0

    return sum(steps_array)/len(steps_array), np.std(steps_array)

def calc_balance_hitting_times():
    sample_size = 1000
    a_est = np.zeros(sample_size)
    x = np.arange(0,20)
    y = np.zeros(len(x))
    yerr = np.zeros(len(x))

        #for k in xrange(sample_size):
    for (ind,i) in enumerate(x[1:]):
        #print "For %d"%i
        avg, std = calc_for_t(i)
        y[ind+1] = avg
        yerr[ind+1] = std

    A = np.vstack([x**2,np.zeros(len(x))]).T
    a,_ = np.linalg.lstsq(A, y)[0]
    #a_est[k] = a

    #    print np.mean(a_est),np.std(a_est)

    if(True):
        c,stats = np.polynomial.polynomial.polyfit(x,y,2,full=True)
        print c, stats

        pplot.figure()
        pplot.errorbar(x,y,yerr=yerr, label='Measured', fmt='o')
        pplot.plot(x,np.polynomial.polynomial.polyval(x,c),label='Polyfit')
        pplot.plot(x,a*x**2, label='LeastSquares on x^2')
        pplot.legend()
        pplot.show()

def calc_irss_for_ten_and_irs_val(tenure,irs_value,mt):
    es = np.random.normal(size=mt)

    val = 0
    irss = []
    irss_at_t = np.zeros(mt)

    for i in xrange(mt):
        irss = [a - 1 for a in irss if a-1 > 0]

        val += es[i]
        if(abs(val)>irs_value):
            val = abs(val) - irs_value
            irss.append(tenure)

        irss_at_t[i] = len(irss)

    return irss_at_t, np.mean(irss_at_t[2*tenure:])

def show_irss_with_unlimited_banks():
    steps = 200000
    tenure = 800
    irs_val = 2
    no_irss, mu = calc_irss_for_ten_and_irs_val(tenure,irs_val,steps)

    pplot.figure()
    pplot.plot(xrange(steps),no_irss, label='Measured')
    pplot.plot(xrange(steps),[tenure/(1.5*(irs_val**2))]*steps, label='Estimated')
    pplot.plot(xrange(steps),[mu]*steps, label='Corrected Mean')
    pplot.legend()
    pplot.show()

def sweep_irss_for_ten_and_val():
    steps = 400000
    x = np.arange(1,11)
    y = np.arange(100,1100,100)
    X, Y = np.meshgrid(x, y)
    zs = np.array([calc_irss_for_ten_and_irs_val(y,x,steps)[1] for x,y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)

    with file('./simulation_data/irs_tenure_no_irs_sweep.bin','wb') as fp:
        pickle.dump(X.tolist(),fp)
        pickle.dump(Y.tolist(),fp)
        pickle.dump(Z.tolist(),fp)


def show_sweep():
    with file('./simulation_data/irs_tenure_no_irs_sweep.bin','rb') as fp:
        x = np.array(pickle.load(fp))
        y = pickle.load(fp)
        z = pickle.load(fp)
    print z

    # IRS
    xr = np.ravel(x)
    # Ten
    yr = np.ravel(y)
    # Func
    zr = np.ravel(z)

    A = np.vstack([yr/(xr**2),np.zeros(len(xr))]).T
    a,_ = np.linalg.lstsq(A, zr)[0]

    zcr = np.array([a*(yri/(xri**2)) for xri,yri in zip(xr,yr)]).reshape(x.shape)


    figure = pplot.figure()
    ax = figure.add_subplot(111, projection='3d')
    ax.set_xlabel('IRS Value')
    ax.set_ylabel('Tenure')
    ax.set_zlabel('Average #IRSs')
    ax.plot_surface(x,y,z,cmap=cm.coolwarm,rstride=1, cstride=1,alpha=0.2)
    ax.plot_surface(x,y,zcr,rstride=1, cstride=1,alpha=0.2)
    pplot.show()

if __name__ == '__main__':
    #show_irss_with_unlimited_banks()
    #calc_balance_hitting_times()
    #sweep_irss_for_ten_and_val()
    show_sweep()
