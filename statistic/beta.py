#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import numpy as np
import scipy.stats as st
from matplotlib import pylab as plt

A = 5.
B = 1.

beta_rv = st.beta(A, B)

MU = A / (A + B)
DISPERSION = (A * B) / (((A + B)**2)*(A+B+1))
SIGMA = math.sqrt(DISPERSION)

print "MU", MU
print "SIGMA", SIGMA

def show_beta_func_graph(N):
    def add_theoretical(N):
        x = np.linspace(0,1,N)
        pdf = beta_rv.pdf(x)
        plt.plot(x, pdf, label='theoretical pdf', c='r', alpha=0.5)

    def add_practical_hist(N, bins=50):
        stats = beta_rv.rvs(N)
        plt.hist(stats, bins, normed=True, facecolor='green', alpha=0.5, label='actual', edgecolor='black', linewidth=.2)

    fig = plt.figure()
    fig.suptitle('Beta Function', fontsize=20)
    plt.ylabel('$f(x)$', fontsize=18)
    plt.xlabel('$x$', fontsize=18)
    add_practical_hist(N, 30)
    add_theoretical(N)
    plt.show()
    plt.clf()
show_beta_func_graph(1000)


def show_CPT_gist(count, k=2, bins=25):
    values = []

    def get_beta_range(N):
        return beta_rv.rvs(N)

    def sample_mean(X):
        return np.mean(X)

    for i in range(count):
        beta_range = get_beta_range(k)
        X = sample_mean(beta_range)
        values.append(X)

    mu = sample_mean(values)
    sigma = np.std(values, ddof=1)
    SE = SIGMA / math.sqrt(k)
    norm_rv = st.norm(MU, SE)

    print 'Count = {:d}, n = {:d}'.format(count,k)
    print 'Теоретическое среднее vs приближенное: {:.2f} vs {:.2f}'.format(MU,mu)
    print 'Теоретическое стандартное откл. vs приближенное: {:.2f} vs {:.2f}'.format(SE,sigma)
    print 'Приближенное среднее 95% доверительный интервал = {:.2f} +/- {:.2f}'.format(mu,2*SE)

    x = np.linspace(0, 1, count)

    plt.xlabel('count='+str(count) + ', n=' + str(k), fontsize=16)
    plt.plot(x, norm_rv.pdf(x), label='theoretical')
    plt.hist(values, bins, normed=True, facecolor='green', alpha=0.5, edgecolor='black', linewidth=.2)
    plt.show()
    plt.clf()


show_CPT_gist(1000, 2)
show_CPT_gist(1000, 5, bins=20)
show_CPT_gist(1000, 10, bins=10)