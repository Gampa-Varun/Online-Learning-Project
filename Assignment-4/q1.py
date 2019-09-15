import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
import random
import bisect
import math

from math import e
#Number of Arms
arm_no = [10,20,30,40,50]

trials = 100

for arms in arm_no:

    epsilon = 0.01
    delt = 0.1


    def kl_div(u,q):#Kl divergence
        if( u!=0.0 ):
            div = u*np.log(1.0*u/(1.0*q)) + (1.0-u)*np.log((1.0-u)/(1.0-q))
        else:
            div = 0
        return div

    def getupperbound(est_mean,t,Nt):
        delta = np.float_power(10,-8)
        eps = np.float_power(10,-5)
        u = est_mean
        if(u <1.0 ):
            if ( u <= delta ):
                u = delta
            q = u + delta

            for i in range(0,2000):
                    f = np.log((4*e+4)*arms*np.float_power(t,2)/delt) + np.log(np.log((4*e+4)*arms*np.float_power(t,2)/delt)) - kl_div(u,q)*Nt
                    df = -1*Nt*(q-u)/(q*(1.0-q))
                    q = np.minimum(((1 - delta)), np.maximum(((q - f/df)),((u + delta))))
                    if(f*f < eps):
                        break

            return q
        else:
            if u == 1:
                return 1


    def getlowerbound(est_mean,t,Nt):
        delta = np.float_power(10,-8)
        eps = np.float_power(10,-5)
        u = est_mean
        if(u >0.0):
            if ( u >= 1 - delta ):
                u = 1 - delta
            q = u - delta
            for i in range(0,2000):
                    f = np.log((4*e+4)*arms*np.float_power(t,2)/delt) + np.log(np.log((4*e+4)*arms*np.float_power(t,2)/delt)) - kl_div(u,q)*Nt

                    df = -1*Nt*(q-u)/(q*(1.0-q))

                    q = np.maximum(((0 + delta)), np.minimum(((q - f/df)),(( u - delta))))

                    if(f*f < eps):


                        break
            return q
        else:
            if u == 0:
                return 0

    error = 0
    T = 0






    for i in range(0,trials):
        t = arms

        mu = np.random.uniform(0.0, 1.0, arms)


        emp_mean = np.random.binomial(1,mu).astype(float)
        N = np.full(arms,1.0)
        u = np.full(arms,1.0)
        l = np.full(arms,0.0)


        for k in np.arange(0,arms):
            u[k] = getupperbound(emp_mean[k],t,N[k])
            l[k] = getlowerbound(emp_mean[k],t,N[k])



        beta =  math.inf
        while beta > epsilon:
            t = t+1

            J = np.argmax(emp_mean)


            Jcompl = np.delete(np.arange(0,arms),J)


            ut = np.argmax(np.take(u,Jcompl))
            ut = Jcompl[ut]
            lt = J

            draw_upper = np.random.binomial(1,mu[ut])
            draw_lower = np.random.binomial(1,mu[lt])

            N[ut] = N[ut] + 1
            N[lt] = N[lt] + 1
            emp_mean[ut] = (emp_mean[ut]*(N[ut] - 1) + draw_upper) /N[ut]
            emp_mean[lt] = (emp_mean[lt]*(N[lt] - 1) + draw_lower) /N[lt]


            u[ut] = getupperbound(emp_mean[ut],t,N[ut])
            l[lt] = getlowerbound(emp_mean[lt],t,N[lt])

            beta = u[ut] - l[lt]

            print(arms,i,beta,t)


        print(i)
        T = T + t

        if(np.argmax(emp_mean)!= np.argmax(mu)):
            error = error+1

    f = open("output.txt", "a")
    print("Average Time or sample complexity for %d number of arms is" %arms, T/trials, file=f)
    print("Error rate in this case is", error/trials,file=f)
