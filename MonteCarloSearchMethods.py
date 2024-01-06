
"""Import these modules to run the functions smoothly

numpy - calculates mathematical functions and helps with arrays
matplotlib - plots points and graphs and functions
math - has mathematical functions
random - implement randomness in MonteCarlo Algorithms
time - compare run_time between functions
stochastic - aids with creating realizations for continuous stochastic processes

***All of the following functions have the primary goal of optimizing an algorithm to find the
minimum of Brownian Motion***
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import random
import time
from stochastic.processes.continuous import CauchyProcess



def f(t, r, s, simT, simR):

    """Function f takes in 5 parameters: t, r, s, simT, and simR.

    (t, simT) - first already simulated Brownian Motion point
    (r, simR) - second already simulated Brownian Motion point
    s - t < s < r

    Outputs ds or the simulated Brownian Motion value at s using a Brownian Bridge
    """

    W = np.random.normal(loc=0, scale=1)

    """W is a simulated standard normal random variable"""

    ds = simR + ((s - r) / (t - r)) * (simT - simR) + np.sqrt(abs(((s - r) * (t - s) / (t - r)))) * W

    """ds is found using a Brownian Bridge formula"""

    return ds


def numberLeast(x):

    """Function numberLeast takes in 1 parameter: x

    x - this function is trying to find the greatest value in the list 'times' that is less than x

    Outputs the index in the list 'times' that is less than x
    ***Helper function for the function BrownianMotionGSS***
    """

    times = np.linspace(0., 1, 2**10 + 1)
    times = list(times)
    count = 0
    for i in range(len(times)):
        if x < times[count]:
            return count - 1
        else:
            pass
        count += 1



def BrownianMotionGSS(n):

    """Function BrownianMotionGSS takes in 1 parameter: x

    n - the number of iterations the function runs the Golden Section Search

    Outputs a visualization of the Golden Section Search on Brownian Motion over n iterations
    """

    T = 1
    y = []
    finaltimes = []
    finaly = []
    times = np.linspace(0., T, 2 ** 10 + 1)
    dB = np.random.normal(loc=0, scale=1, size=(1))

    """Simulates the Brownian Motion Path using the Brownian Bridge formulas"""
    for i in range(2 ** 10 + 1):
        if i == 0:
            y.append(0)
            pass
        elif i == 2 ** 10:
            pass
        elif i == 1:
            Z = np.random.normal(loc=0, scale=1)
            s = times[1]
            dr = s * dB + np.sqrt(s * (T - s)) * Z
            y.append(dr[0])
        else:
            s = times[i]
            r = times[i - 1]
            W = np.random.normal(loc=0, scale=1)
            ds = dr + ((s - r) / (T - r)) * (dB - dr) + np.sqrt(((s - r) * (T - s) / (T - r))) * W
            y.append(ds[0])
            dr = ds
    times = list(times)
    times.pop(0)
    times.pop(2**10-1)
    y = list(y)
    y.pop(0)



    """Runs the Golden Section Search for 'n' iterations"""
    lt0 = []
    lt3 = []
    ft0 = []
    ft3 = []
    goldenRatio = (1 + (math.sqrt(5))) / 2
    t0 = 0
    f0 = 0
    t3 = 1
    f3 = dB[0]
    t2 = t0 + ((t3 - t0) / goldenRatio)
    t1 = t0 + ((t3 - t0) / (goldenRatio ** 2))
    w = numberLeast(t1)
    v = numberLeast(t2)
    c = f(T, times[w], t1, t3, y[w])
    d = f(T, times[v], t2, t3, y[v])
    lt0.append(t0)
    ft0.append(f0)
    lt3.append(t3)
    ft3.append(f3)
    finaltimes.append(t0)
    finaltimes.append(t1)
    finaltimes.append(t2)
    finaltimes.append(t3)
    finaly.append(f0)
    finaly.append(c)
    finaly.append(d)
    finaly.append(f3)
    for i in range(n-1):
        if d > c:
            t3 = t2
            v = numberLeast(t2)
            f3 = f(times[v+1], times[v], t2, y[v+1], y[v])
            t2 = t1
            d = c
            t1 = (t0 + ((t3 - t0) / (goldenRatio ** 2)))
            w = numberLeast(t1)
            c = f(times[w+1], times[w], t1, y[w+1], y[w])
            lt3.append(t3)
            ft3.append(f3)
            finaltimes.append(t1)
            finaly.append(c)
            finaltimes.append(t3)
            finaly.append(f3)
        else:
            t0 = t1
            w = numberLeast(t1)
            f0 = f(times[w+1], times[w], t1, y[w+1], y[w])
            t1 = t2
            c = d
            t2 = t0 + ((t3 - t0) / goldenRatio)
            v = numberLeast(t2)
            d = f(times[v+1], times[v], t2, y[v+1], y[v])
            lt0.append(t0)
            ft0.append(f0)
            finaltimes.append(t2)
            finaly.append(d)
            finaltimes.append(t0)
            finaly.append(f0)


    """Plots the visualized Golden Section Search on the simulated Brownian Motion Path"""
    plt.plot(lt0, ft0, marker='.')
    plt.plot(lt3, ft3, marker='.')
    times = finaltimes + times
    y = finaly + y
    C = np.array(times)
    D = np.array(y)
    plt.plot(sorted(C), D[np.argsort(C)], 'black')
    plt.show()



def iterativeBrownianMotionGSS(n, epsilon, l):


    """Function iterativeBrownianMotionGSS takes in 3 parameters: n, epsilon, l

    n - the number of iterations the function runs the Golden Section Search for each partitioned interval
    epsilon - specifies a threshold for which the Golden Section Search terminates depending on the change in y between two subsequent iterations
    l - 2**l is the number of partitions that the interval [0,1] is split into

    Outputs a visualization of the Iterated Golden Section Search on Brownian Motion over n iterations per partition, with
    graphs of a Brownian Motion GSS and a graph of the two minima found from the aforementioned algorithms on the simulated BM path.
    Also outputs a tuple that contains both minima found from the Iterated GSS and GSS

    ***The 'Iterated Golden Section Search' essentially partitions an interval and runs a GSS on each of these partitioned intervals***
    """


    color_list = ['orange', 'red', 'blue','cyan','pink','purple','green','dodgerblue']
    min_list = []
    min_t_list = []
    T = 1
    line = np.linspace(0,T, 2**l+1)
    y = []
    times = np.linspace(0., T, 2 ** 10 + 1)
    finaly = []
    finaltimes = []
    dB = np.random.normal(loc=0, scale=1, size=(1))


    """Simulates the Brownian Motion Path using the Brownian Bridge formulas"""
    for i in range(2 ** 10 + 1):
        if i == 0:
            y.append(0)
            pass
        elif i == 2 ** 10:
            pass
        elif i == 1:
            Z = np.random.normal(loc=0, scale=1)
            s = times[1]
            dr = s * dB + np.sqrt(s * (T - s)) * Z
            y.append(dr[0])
        else:
            s = times[i]
            r = times[i - 1]
            W = np.random.normal(loc=0, scale=1)
            ds = dr + ((s - r) / (T - r)) * (dB - dr) + np.sqrt(((s - r) * (T - s) / (T - r))) * W
            y.append(ds[0])
            dr = ds
    y = [0] + y


    """This for loop partitions the interval [0,1] into 2**l sections and runs the GSS for each of these sections"""
    for j in range(2**l):


        """The following code runs the GSS for each individual partitioned sections"""
        if j == 0:
            t0 = 0
            f0 = 0
            times = list(times)
            t3 = line[j+1]
            f3 = y[times.index(line[j+1])]
        else:
            t0 = line[j]
            f0 = y[times.index(line[j])]
            t3 = line[j+1]
            f3 = y[times.index(line[j+1])]
        lt0 = []
        lt3 = []
        ft0 = []
        ft3 = []
        goldenRatio = (1 + (math.sqrt(5))) / 2
        t2 = t0 + ((t3 - t0) / goldenRatio)
        t1 = t0 + ((t3 - t0) / (goldenRatio ** 2))
        w = numberLeast(t1)
        v = numberLeast(t2)
        c = f(times[w+1], times[w], t1, t3, y[w])
        d = f(times[v+1], times[v], t2, t3, y[v])
        lt0.append(t0)
        ft0.append(f0)
        lt3.append(t3)
        ft3.append(f3)
        finaltimes.append(t1)
        finaltimes.append(t2)
        finaly.append(c)
        finaly.append(d)
        for i in range(n-1):
            if d > c:
                t3 = t2
                v = numberLeast(t2)
                if abs(f3 - f(times[v + 1], times[v], t2, y[v + 1], y[v])) < epsilon:
                    break
                f3 = f(times[v + 1], times[v], t2, y[v + 1], y[v])
                t2 = t1
                d = c
                t1 = (t0 + ((t3 - t0) / (goldenRatio ** 2)))
                w = numberLeast(t1)
                c = f(times[w + 1], times[w], t1, y[w + 1], y[w])
                lt3.append(t3)
                ft3.append(f3)
                finaltimes.append(t1)
                finaly.append(c)
                finaltimes.append(t3)
                finaly.append(f3)
            else:
                t0 = t1
                w = numberLeast(t1)
                if abs(f0 - f(times[w + 1], times[w], t2, y[w + 1], y[w])) < epsilon:
                    break
                f0 = f(times[w + 1], times[w], t1, y[w + 1], y[w])
                t1 = t2
                c = d
                t2 = t0 + ((t3 - t0) / goldenRatio)
                v = numberLeast(t2)
                d = f(times[v + 1], times[v], t2, y[v + 1], y[v])
                lt0.append(t0)
                ft0.append(f0)
                finaltimes.append(t2)
                finaly.append(d)
                finaltimes.append(t0)
                finaly.append(f0)

        """The following code appends the minimum found by each 'sub-minimum' found on each GSS and graphs the visualization"""
        min_list.append((f0 + f3)/2)
        min_t_list.append((t0 + t3) / 2)
        plt.plot(lt0, ft0, marker='.', color=color_list[j%8])
        plt.plot(lt3, ft3, marker='.', color=color_list[j%8])
        plt.title(f'Iterated Brownian Motion GSS Over {2**l} Intervals')



    """The following code runs a regular Brownian Motion GSS and graphs the visual for comparison"""
    min_list.append(0)
    min_t_list.append(0)
    min_t_list.append(1)
    min_list.append(y[times.index(line[2 ** l])])
    plt.show()
    line = np.linspace(0, T, 2 ** 0 + 1)
    finaly = []
    finaltimes = []
    for j in range(2 ** 0):
        if j == 0:
            t0 = 0
            f0 = 0
            times = list(times)
            t3 = line[j + 1]
            f3 = y[times.index(line[j + 1])]
        else:
            t0 = line[j]
            f0 = y[times.index(line[j])]
            t3 = line[j + 1]
            f3 = y[times.index(line[j + 1])]
        lt0 = []
        lt3 = []
        ft0 = []
        ft3 = []
        goldenRatio = (1 + (math.sqrt(5))) / 2
        t2 = t0 + ((t3 - t0) / goldenRatio)
        t1 = t0 + ((t3 - t0) / (goldenRatio ** 2))
        w = numberLeast(t1)
        v = numberLeast(t2)
        c = f(times[w + 1], times[w], t1, t3, y[w])
        d = f(times[v + 1], times[v], t2, t3, y[v])
        lt0.append(t0)
        ft0.append(f0)
        lt3.append(t3)
        ft3.append(f3)
        finaltimes.append(t1)
        finaltimes.append(t2)
        finaly.append(c)
        finaly.append(d)
        for i in range(100):
            if d > c:
                t3 = t2
                v = numberLeast(t2)
                if abs(f3 - f(times[v + 1], times[v], t2, y[v + 1], y[v])) < epsilon:
                    break
                f3 = f(times[v + 1], times[v], t2, y[v + 1], y[v])
                t2 = t1
                d = c
                t1 = (t0 + ((t3 - t0) / (goldenRatio ** 2)))
                w = numberLeast(t1)
                c = f(times[w + 1], times[w], t1, y[w + 1], y[w])
                lt3.append(t3)
                ft3.append(f3)
                finaltimes.append(t1)
                finaly.append(c)
                finaltimes.append(t3)
                finaly.append(f3)
            else:
                t0 = t1
                w = numberLeast(t1)
                if abs(f0 - f(times[w + 1], times[w], t2, y[w + 1], y[w])) < epsilon:
                    break
                f0 = f(times[w + 1], times[w], t1, y[w + 1], y[w])
                t1 = t2
                c = d
                t2 = t0 + ((t3 - t0) / goldenRatio)
                v = numberLeast(t2)
                d = f(times[v + 1], times[v], t2, y[v + 1], y[v])
                lt0.append(t0)
                ft0.append(f0)
                finaltimes.append(t2)
                finaly.append(d)
                finaltimes.append(t0)
                finaly.append(f0)
        plt.plot(times, y, color='black')
        plt.plot(lt0, ft0, marker='.', color=color_list[j % 8])
        plt.plot(lt3, ft3, marker='.', color=color_list[j % 8])
        plt.title('Brownian Motion GSS')
        plt.show()

        """The subsequent code plots the Brownian Motion Path itself with the minima found by the GSS and Iterated GSS"""
        plt.title('Simulated Brownian Motion Path')
        plt.plot(times, y, color='black')
        plt.plot(min_t_list[min_list.index(min(min_list))], min(min_list), marker='o', color='magenta',
                 label=f'Iterated GSS Minimum\n({round(min_t_list[min_list.index(min(min_list))], 4)}, {round(min(min_list), 4)})')
        plt.plot((t0 + t3) / 2, (f0 + f3) / 2, marker='o', color='aquamarine',
                 label=f'Naive GSS Minimum\n({round((t0 + t3) / 2, 4)}, {round((f0 + f3) / 2, 4)})')
        leg = plt.legend()
        plt.show()


    return (min(min_list), ((f0+f3)/2 - min(y)))

def iterativeBrownianMotionGSSReturnOnly(k, n, epsilon, l):

    """Function iterativeBrownianMotionGSSReturnOnly takes in 4 parameters: k, n, epsilon, l

    k - 2^k+1 amount of simulated Brownian Motion points
    n - the number of iterations the function runs the Golden Section Search for each partitioned interval
    epsilon - specifies a threshold for which the Golden Section Search terminates depending on the change in y between two subsequent iterations
    l - 2**l is the number of partitions that the interval [0,1] is split into

    Similar to the iterativeBrownianMotion function, but it only outputs the positive difference between the two minima found using
    the Iterated GSS and Brownian Motion GSS

    ***The 'Iterated Golden Section Search' essentially partitions an interval and runs a GSS on each of these partitioned intervals***
    """

    color_list = ['orange', 'red', 'blue','cyan','pink','purple','green','dodgerblue']
    min_list = []
    min_t_list = []
    T = 1
    line = np.linspace(0,T, 2**l+1)
    y = []
    times = np.linspace(0., T, 2 ** k + 1)
    finaly = []
    finaltimes = []
    dB = np.random.normal(loc=0, scale=1, size=(1))

    """Simulates the Brownian Motion Path for 2^k + 1 values"""
    for i in range(2 ** k + 1):
        if i == 0:
            y.append(0)
            pass
        elif i == 2 ** k:
            pass
        elif i == 1:
            Z = np.random.normal(loc=0, scale=1)
            s = times[1]
            dr = s * dB + np.sqrt(s * (T - s)) * Z
            y.append(dr[0])
        else:
            s = times[i]
            r = times[i - 1]
            W = np.random.normal(loc=0, scale=1)
            ds = dr + ((s - r) / (T - r)) * (dB - dr) + np.sqrt(((s - r) * (T - s) / (T - r))) * W
            y.append(ds[0])
            dr = ds
    y = [0] + y

    """This for loop partitions the interval [0,1] into 2**l sections and runs the GSS for each of these sections"""
    for j in range(2**l):

        """The following code runs the iterative GSS over the 2**l partitions"""
        if j == 0:
            t0 = 0
            f0 = 0
            times = list(times)
            t3 = line[j+1]
            f3 = y[times.index(line[j+1])]
        else:
            t0 = line[j]
            f0 = y[times.index(line[j])]
            t3 = line[j+1]
            f3 = y[times.index(line[j+1])]
        lt0 = []
        lt3 = []
        ft0 = []
        ft3 = []
        goldenRatio = (1 + (math.sqrt(5))) / 2
        t2 = t0 + ((t3 - t0) / goldenRatio)
        t1 = t0 + ((t3 - t0) / (goldenRatio ** 2))
        w = numberLeast(t1)
        v = numberLeast(t2)
        c = f(times[w+1], times[w], t1, t3, y[w])
        d = f(times[v+1], times[v], t2, t3, y[v])
        lt0.append(t0)
        ft0.append(f0)
        lt3.append(t3)
        ft3.append(f3)
        finaltimes.append(t1)
        finaltimes.append(t2)
        finaly.append(c)
        finaly.append(d)
        for i in range(n):
            if d > c:
                t3 = t2
                v = numberLeast(t2)
                if abs(f3 - f(times[v + 1], times[v], t2, y[v + 1], y[v])) < epsilon:
                    break
                f3 = f(times[v + 1], times[v], t2, y[v + 1], y[v])
                t2 = t1
                d = c
                t1 = (t0 + ((t3 - t0) / (goldenRatio ** 2)))
                w = numberLeast(t1)
                c = f(times[w + 1], times[w], t1, y[w + 1], y[w])
                lt3.append(t3)
                ft3.append(f3)
                finaltimes.append(t1)
                finaly.append(c)
                finaltimes.append(t3)
                finaly.append(f3)
            else:
                t0 = t1
                w = numberLeast(t1)
                if abs(f0 - f(times[w + 1], times[w], t2, y[w + 1], y[w])) < epsilon:
                    break
                f0 = f(times[w + 1], times[w], t1, y[w + 1], y[w])
                t1 = t2
                c = d
                t2 = t0 + ((t3 - t0) / goldenRatio)
                v = numberLeast(t2)
                d = f(times[v + 1], times[v], t2, y[v + 1], y[v])
                lt0.append(t0)
                ft0.append(f0)
                finaltimes.append(t2)
                finaly.append(d)
                finaltimes.append(t0)
                finaly.append(f0)
        min_list.append((f0 + f3)/2)
        min_t_list.append((t0 + t3) / 2)

    """We take into account the simulated BM values at t = 0 and t = 1 because sometimes the GSS misses these minima"""
    min_list.append(0)
    min_t_list.append(0)
    min_t_list.append(1)
    min_list.append(y[times.index(line[2 ** l])])

    """Delta is defined to compare the difference between the minima found using the Iterated GSS and the actual BM minimum"""
    delta = abs(min(min_list) - min(y))
    return delta




def GSSwithBis(n, l, k):

    """Function GSSWithBis takes in 3 parameters: n, l, k

    n - 2^n + 1 simulated BM points to create a simulated BM path
    l - number of iterated partitions of the interval [0,1] (i.e., if l = 2, we partition the interval into 2 sub-intervals, choose an interval, repeating
    this process 2 times)
    k - the number of selected intervals, which each are obtained through l iterated partitions

    GSSWithBis is another modified version of the GSS function. It first iteratively partitions the interval [0,1] until
    it reaches certain small interval and estimates the minimum by finding the average of the y-coordinates. It continues to do this
    for k intervals. Eventually, the final estimated minimum is the minimum of the list of estimated minima found over the k intervals.
    This function is a MonteCarlo Algorithm.

    Specifically, this function outputs a visualization of this method in addition to returning a tuple containing the actual minimum
    and the estimated minimum.
    """

    color_list = ['orange', 'red', 'blue', 'cyan', 'pink', 'purple', 'green', 'dodgerblue']
    T = 1
    min_list = []
    min_t_list = []
    times = np.linspace(0., T, 2 ** n + 1)
    dB = np.random.normal(loc=0, scale=1, size=(1))
    y = []
    goldenRatio = (1 + (math.sqrt(5))) / 2


    """The following code simulates a Brownian Motion path with 2^n + 1 values"""
    for i in range(2 ** n + 1):
        if i == 0:
            y.append(0)
            pass
        elif i == 2 ** n:
            pass
        elif i == 1:
            Z = np.random.normal(loc=0, scale=1)
            s = times[1]
            dr = s * dB + np.sqrt(s * (T - s)) * Z
            y.append(dr[0])
        else:
            s = times[i]
            r = times[i - 1]
            W = np.random.normal(loc=0, scale=1)
            ds = dr + ((s - r) / (T - r)) * (dB - dr) + np.sqrt(((s - r) * (T - s) / (T - r))) * W
            y.append(ds[0])
            dr = ds
    y.append(dB[0])


    """The following for loop chooses k intervals randomly, each obtained through l iterated paritions"""
    for j in range(k):
        subinttimes = times
        subinty = y


        """The following code iteratively partitions the interval [0,1] for l iterations"""
        for i in range(l):
            rint = random.randint(0, 1)

            """The following if-else sequence essentially randomly chooses a subinterval to keep partitioning until l iterations are reached"""
            if rint == 0:
                subinttimes = subinttimes[0:len(subinttimes) // 2]
                subinty = subinty[0:(len(subinty) // 2) + 1]
            elif rint == 1:
                subinttimes = subinttimes[len(subinttimes) // 2: len(subinttimes)]
                subinty = subinty[len(subinty) // 2: len(subinty)]


        """The following code appends the estimated minima from the bisection method and plots these selected intervals and estimated minima"""
        #min_list.append(-expectedMaxSpecific(subinty[len(subinty) - 1], subinty[0], subinttimes[len(subinttimes)-1], subinttimes[0]))
        min_t_list.append((subinttimes[0] + subinttimes[len(subinttimes)-1])/2)
        min_list.append((subinty[0] + subinty[len(subinty)-1])/2)
        plt.plot(subinttimes[0], subinty[0], color = 'red', marker='o')
        plt.plot((subinttimes[0] + subinttimes[len(subinttimes)-1])/2, (subinty[0] + subinty[len(subinty)-1])/2, color='blue', marker='o')
        plt.plot(subinttimes[len(subinttimes)-1], subinty[len(subinty)-1], color='red', marker='o')
        #plt.plot(0, -expectedMaxSpecific(subinty[len(subinty) - 1], subinty[0], subinttimes[len(subinttimes)-1], subinttimes[0]), marker='o')

    """This code plots the simulated Brownian Motion Path"""
    plt.title(f"MCB with $l$ = {n}, $r$ = {l}, and $g$ = {k}")
    plt.plot(times,y,color='black')
    plt.show()

    plt.title('Simulated Brownian Motion Path')
    plt.plot(times, y, color='black')
    plt.plot(min_t_list[min_list.index(min(min_list))], min(min_list), marker='o', color='magenta',
             label=f'MCB Minimum\n({round(min_t_list[min_list.index(min(min_list))], 4)}, {round(min(min_list), 4)})')
    plt.plot(times[y.index(min(y))], min(y), marker='o', color='aquamarine',
             label=f'Actual Global Minimum\n({round(times[y.index(min(y))], 4)}, {round(min(y), 4)})')
    leg = plt.legend()
    plt.show()

    """Returns the tuple containing the actual minimum and estimated minimum via the GSSwithBis algorithm"""
    return (min(min_list), min(y))

def GSSwithBisReturnOnly(n, l, k):


    """Function GSSWithBisReturn takes in 3 parameters: n, l, k

    n - 2^n + 1 simulated BM points to create a simulated BM path
    l - number of iterated partitions of the interval [0,1] (i.e., if l = 2, we partition the interval into 2 sub-intervals, choose an interval, repeating
    this process 2 times)
    k - the number of selected intervals, which each are obtained through l iterated partitions

    This function is very similar to the GSSwithBis function, but it only returns a tuple containing the positive difference between the
    estimated minimum and the actual minimum in addition to the estimated minimum
    """


    T = 1
    min_list = []
    min_t_list = []
    times = np.linspace(0., T, 2 ** n + 1)
    dB = np.random.normal(loc=0, scale=1, size=(1))
    y = []
    goldenRatio = (1 + (math.sqrt(5))) / 2


    """The following code simulates a Brownian Motion path with 2^n + 1 values"""
    for i in range(2 ** n + 1):
        if i == 0:
            y.append(0)
            pass
        elif i == 2 ** n:
            pass
        elif i == 1:
            Z = np.random.normal(loc=0, scale=1)
            s = times[1]
            dr = s * dB + np.sqrt(s * (T - s)) * Z
            y.append(dr[0])
        else:
            s = times[i]
            r = times[i - 1]
            W = np.random.normal(loc=0, scale=1)
            ds = dr + ((s - r) / (T - r)) * (dB - dr) + np.sqrt(((s - r) * (T - s) / (T - r))) * W
            y.append(ds[0])
            dr = ds
    y.append(dB[0])


    """The following for loop chooses k intervals randomly, each obtained through l iterated paritions"""
    for j in range(k):
        subinttimes = times
        subinty = y


        """The following code iteratively partitions the interval [0,1] for l iterations"""
        for i in range(l):
            rint = random.randint(0, 1)


            """The following if-else sequence essentially randomly chooses a subinterval to keep partitioning until l iterations are reached"""
            if rint == 0:
                subinttimes = subinttimes[0:len(subinttimes) // 2]
                subinty = subinty[0:(len(subinty) // 2) + 1]
            elif rint == 1:
                subinttimes = subinttimes[len(subinttimes) // 2: len(subinttimes)]
                subinty = subinty[len(subinty) // 2: len(subinty)]
        min_t_list.append((subinttimes[0] + subinttimes[len(subinttimes)-1])/2)
        min_list.append((subinty[0] + subinty[len(subinty)-1])/2)


    """Returns the tuple containing the difference between the estimated and actual minimum in addition to
    the estimated minimum found via the GSSwithBis algorithm"""
    return (abs(min(min_list)-min(y)), min(min_list))




def comparisonBisVsGSS(n, l, epsilon, k):
    """Function iterativeBrownianMotionGSSReturnOnly takes in 4 parameters: n, l, epsilon, k

    k - 2**k amount of simulated Brownian Motion points
    n - the number of iterations the function runs the Golden Section Search for each partitioned interval/used as input for each of n, l, and
    k parameters in the GSS with Bisection algorithm
    epsilon - specifies a threshold for which the Golden Section Search depending on the change in y between two subsequent iterations
    l - 2**l is the number of partitions that the interval [0,1] is split into

    Prints the execution time of the bisection algorithm, iterative GSS, and the comparison between the two for the same approximate input size
    """

    """The subsequent code calculates the execution time for the Bisection algorithm"""
    start_time = time.perf_counter()
    GSSwithBisReturnOnly(n, n, n)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    bisexec = execution_time
    print(f"The execution time of the Bisection is: {execution_time}")

    """The subsequent code calculates the execution time for the Interative GSS algorithm"""
    start_time = time.perf_counter()
    iterativeBrownianMotionGSSReturnOnly(n, l, epsilon, k)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    itexec = execution_time
    print(f"The execution time of the Iterative GSS is: {execution_time}")

    """Prints how much faster the Bisection method is compared to the Iterative GSS algorithm"""
    print(f"The execution time of the Bisection is {itexec/bisexec} times faster than the Iterative GSS")


def bisectionOnCauchy(n, l, k):

    """Function bisectionOnCauchy takes in 3 parameters: n, l, k

        n - 2^n + 1 simulated BM points to create a simulated BM path
        l - number of iterated partitions of the interval [0,1] (i.e., if l = 2, we partition the interval into 2 sub-intervals, choose an interval, repeating
        this process 2 times)
        k - the number of selected intervals, which each are obtained through l iterated partitions

        bisectionOnCauchy is a modified version of the GSSwithBis function, but it runs on the Cauchy Process. It first iteratively partitions the interval
        [0,1] until it reaches certain small interval and estimates the minimum by finding the average of the y-coordinates. It continues to do this
        for k intervals. Eventually, the final estimated minimum is the minimum of the list of estimated minima found over the k intervals.
        This function is a MonteCarlo Algorithm.

        Specifically, this function outputs a visualization of this method in addition to returning a tuple containing the actual minimum
        and the estimated minimum.
        """

    """Creates a Cauchy Process object so that points may be simulated"""
    cp = CauchyProcess(t=1, rng=None)

    """Partitions the [0,1] interval into 2**n smaller points at which Cauchy process  points are simulated"""
    times = np.linspace(0., 1, 2 ** n + 1)
    y = cp.sample(2**n)


    min_list = []
    min_t_list = []


    """The following for loop chooses k intervals randomly, each obtained through l iterated paritions"""
    for j in range(k):
        subinttimes = times
        subinty = y

        """The following code iteratively partitions the interval [0,1] for l iterations"""
        for i in range(l):
            rint = random.randint(0, 1)

            """The following if-else sequence essentially randomly chooses a subinterval to keep partitioning until l iterations are reached"""
            if rint == 0:
                subinttimes = subinttimes[0:len(subinttimes) // 2]
                subinty = subinty[0:(len(subinty) // 2) + 1]
            elif rint == 1:
                subinttimes = subinttimes[len(subinttimes) // 2: len(subinttimes)]
                subinty = subinty[len(subinty) // 2: len(subinty)]

        """The following code appends the estimated minima from the bisection method and plots these selected intervals and estimated minima"""
        # min_list.append(-expectedMaxSpecific(subinty[len(subinty) - 1], subinty[0], subinttimes[len(subinttimes)-1], subinttimes[0]))
        min_t_list.append((subinttimes[0] + subinttimes[len(subinttimes) - 1]) / 2)
        min_list.append((subinty[0] + subinty[len(subinty) - 1]) / 2)
        plt.plot(subinttimes[0], subinty[0], color='red', marker='o')
        plt.plot((subinttimes[0] + subinttimes[len(subinttimes) - 1]) / 2, (subinty[0] + subinty[len(subinty) - 1]) / 2,
                 color='blue', marker='o')
        plt.plot(subinttimes[len(subinttimes) - 1], subinty[len(subinty) - 1], color='red', marker='o')
        # plt.plot(0, -expectedMaxSpecific(subinty[len(subinty) - 1], subinty[0], subinttimes[len(subinttimes)-1], subinttimes[0]), marker='o')

    """This code plots the simulated Cauchy Process Path"""
    plt.title(f"MCB with l = {n}, r = {l}, and g = {k}")
    plt.plot(times, y, color='black')
    plt.show()
    y = list(y)

    plt.title('Simulated Cauchy Process Path')
    plt.plot(times, y, color='black')
    plt.plot(min_t_list[min_list.index(min(min_list))], min(min_list), marker='o', color='magenta',
             label=f'MCB Minimum\n({round(min_t_list[min_list.index(min(min_list))], 4)}, {round(min(min_list), 4)})')
    plt.plot(times[y.index(min(y))], min(y), marker='o', color='aquamarine',
             label=f'Actual Global Minimum\n({round(times[y.index(min(y))], 4)}, {round(min(y), 4)})')
    leg = plt.legend()
    plt.show()

    """Returns the tuple containing the actual minimum and estimated minimum via the GSSwithBis algorithm"""
    return (min(min_list), min(y))




