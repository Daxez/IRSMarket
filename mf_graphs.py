"""
Some graphs that I wanted to see plotted (for mean field/analytical approach.)
"""

from __future__ import division

import numpy as np
import math
import scipy.integrate as integrate
import matplotlib.pyplot as pplot
from mpl_toolkits.mplot3d import Axes3D

from scipy.special import erf

def first_hit_distribution(threshold,t):
    return threshold/(math.sqrt(2*math.pi*(t**3))) * math.exp(-1*(threshold**2)/(2*t))

def first_hit_cum(T,t):
    return 1-(math.sqrt(t**3) * erf(T/(math.sqrt(2*t))))/pow(t,3/2)

def smallest_hit(T,n,t):
    return first_hit_distribution(T,t)*n*(1-first_hit_cum(T,t))**(n-1)

def actual_p(T,k,t,a=0.3):
    ki = 1-((1/k)*(1-a))

    diff = erf(T/(math.sqrt(2)*t)) - erf(ki*T/(math.sqrt(2)*t))
    mg = erf(T/(math.sqrt(2)*t))

    return diff/mg

def p(T,k,n,t,a):
    return smallest_hit(T,n,t) * actual_p(T,k,t,a)
    #return first_hit_distribution(T,t) * actual_p(T,k,t)

def plot_actual():
    T = 20
    k = 5

    fig = pplot.figure()
    ax = fig.add_subplot(1,1,1)
    rng = range(1,100)
    y = [actual_p(T,k,t) for t in rng]
    ax.plot(rng,y)

    pplot.show()

def integrate_and_plot(threshold,degree,n,a):
    I = integrate.quad(lambda x: p(threshold,degree,n,x,a), 0.01, np.inf)
    print I

    ## Plot

    fig = pplot.figure()
    ax = fig.add_subplot(3,1,1)

    x = np.linspace(1,500,5000)
    y = [p(threshold,degree,n, t,a) for t in x]

    ax.plot(x,y)
    title = "f(t)*p(t) Integrated: %f, k: %d, T: %d"%(I[0],degree,threshold)
    ax.set_title(title)

    ax = fig.add_subplot(3,1,2)

    y = [actual_p(threshold,degree, t,a) for t in x]

    ax.plot(x,y)
    ax.set_title("p(t): P(-T > t > -(1-1/k T)")

    ax = fig.add_subplot(3,1,3)

    y = [smallest_hit(threshold,n,t) for t in x]

    ax.plot(x,y)
    ax.set_title("f(t): First hit density")

    pplot.show()

def plot_branching_ratio():
    def fun(x,y):
        return y*integrate.quad(lambda t: p(x,y,n,t), 0.01, np.inf)[0]

    fig = pplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.arange(1, 100, 1)
    X, Y = np.meshgrid(x, y)
    zs = np.array([fun(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)

    ax.plot_surface(X, Y, Z)

    ax.set_xlabel('Threshold')
    ax.set_ylabel('Degree')
    ax.set_zlabel('Branching ratio')

    pplot.show()


def plot_branching_ratio_K_N():
    threshold = 50

    def fun(x,y):
        return y*integrate.quad(lambda t: p(threshold,y,x,t), 0.01, np.inf)[0]

    fig = pplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.arange(1, 100, 1)
    X, Y = np.meshgrid(x, y)
    zs = np.array([fun(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)

    ax.plot_surface(X, Y, Z)

    ax.set_xlabel('N')
    ax.set_ylabel('Degree')
    ax.set_zlabel('Branching ratio')

    pplot.show()

def mt(k):
    fig = pplot.figure()
    ax = fig.add_subplot(111)

    x = np.linspace(1,100,100)
    #y = x**(-3/2)*(1/k * (1-(1/k)))**x
    y = np.exp(x*np.log((4/k)*(1-(1/k))))

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.plot(x,y)
    pplot.show()

def plot_integrand_by_alpha(T,k,n):
    fig = pplot.figure()
    ax = fig.add_subplot(111)

    a = np.linspace(0,1,100)
    y = [integrate.quad(lambda x: p(T,k,n,x,ai), 0.001, np.inf)[0] for ai in a]

    ax.plot(a, y)
    pplot.show()


if __name__ == "__main__":
    T = 20
    k = 30
    n = 10
    a = 0.1

    #mt(k)

    #integrate_and_plot(T,k,n,a)
    #plot_branching_ratio()
    plot_integrand_by_alpha(T,k,n)
