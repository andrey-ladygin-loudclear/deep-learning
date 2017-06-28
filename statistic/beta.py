import math
import scipy.stats as st
from matplotlib import pylab as plt

x_coordinates = []
y_coordinates = []

A = 5.
B = 1.
N = 10000


t = (3**5)

mathematical_expectation = A / (A + B)
dispersion = A * B / (A + B)**2 * (A + B + 1)

print "mathematical_expectation", mathematical_expectation
print "dispersion", dispersion

def BetaFn(x):
    return st.beta.pdf(x, A, B)

def show_beta_func_graph():
    for x in range(N):
        x = float(x) / N
        y = BetaFn(x)
        # print "X = %s, Y = %s" % (x, y)
        x_coordinates.append(x)
        y_coordinates.append(y)

    fig = plt.figure()
    fig.suptitle('Beta Function', fontsize=20)
    plt.xlabel('N', fontsize=18)
    plt.ylabel('Value', fontsize=16)

    stats = st.beta.rvs(A, B, size=N)
    plt.hist(stats, 100, normed=True, facecolor='green', alpha=0.5)
    plt.plot(x_coordinates, y_coordinates)
    plt.show()

    plt.clf()

def get_beta_range(N):
    return st.beta.rvs(A, B, size=N)

def x_sample_mean(X):
    return sum(X) / len(X)

def show_CPT_gist(m, k=2):
    values = []
    plot_x = []

    for i in range(m):
        beta_range = get_beta_range(k)
        X = x_sample_mean(beta_range)
        values.append(X)
        #plot_x.append(X - mathematical_expectation / (math.sqrt(float(dispersion)) / math.sqrt(float(k))))

    plt.xlabel('count='+str(m) + ', n=' + str(k), fontsize=16)
    #plt.plot(plot_x)
    plt.hist(values, 70, normed=True, facecolor='green', alpha=0.5)
    plt.show()
    plt.clf()

show_beta_func_graph()
# show_CPT_gist(1000, 2)
# show_CPT_gist(1000, 5)
# show_CPT_gist(1000, 10)
# show_CPT_gist(1000, 50)
# show_CPT_gist(1000, 80)
show_CPT_gist(1000, 200)
show_CPT_gist(10000, 200)
