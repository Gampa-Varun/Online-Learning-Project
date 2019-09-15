import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
import random
import bisect
import math

from math import e
#Number of Arms
arm_no = [3,10,20,30,40,50]
trials = 100
k = 0
T = 0
error = 0

for arms in arm_no:
    for i in range(0,trials):

        epsilon = 0.01
        t = 1
        delta = 0.1
        lbd = 9
        beta = 1
        sigma = 0.5

        mu = np.random.uniform(0.0, 1.0, arms)

        cont = True

        emp_mean = np.random.binomial(1,mu).astype(float)
        N = np.full(arms,1.0)

        t = 0

        while(cont):
            ubounds = (emp_mean + (1+beta)*(1+np.sqrt(epsilon))*np.sqrt(2*sigma*sigma*(1+epsilon)*np.log(np.log((1+epsilon)*N)/delta)/N))

            It = np.argmax(ubounds)
            N[It] = N[It] + 1
            draw = np.random.binomial(1,mu[It])
            emp_mean[It] = (emp_mean[It]*(N[It]-1.0) + 1.0*draw)/N[It]
            print(arms,i,t)
            t = t+1

            for k in range(0,arms):
                if(N[k] <1 + lbd*np.sum(N) - lbd*N[k]  ):
                    cont = True
                else:

                    cont = False
                    break
        T= t + T
        k = k+1

        if(np.argmax(emp_mean)!= np.argmax(mu)):
            error = error+1

    f = open("output2.txt", "a")
    print("Average Time or sample complexity for %d number of arms is" %arms, T/trials, file=f)
    print("Error rate in this case is", error/trials,file=f)
