import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
import random
import bisect

#Number of Arms
k = 10
#Number of rounds
T = np.arange(0,25000)

def kl_div(u,q):#Kl divergence
    if( u!=0.0 ):
        div = u*np.log(1.0*u/(1.0*q)) + (1.0-u)*np.log((1.0-u)/(1.0-q))
    else:
        div = 0
    return div

def getupperbound(est_mean,t):
    delta = np.float_power(10,-8)
    eps = np.float_power(10,-12)
    u = est_mean[0]
    if(u <1.0):
        if ( u <= delta ):
            u = delta
        q = u + delta
        for i in range(0,20):
                f = np.log(t) - kl_div(u,q)*est_mean[1]
                df = -(q-u)/(q*(1.0-q))
                q = np.minimum(((1 - delta)), np.maximum(((q - f/df)),(( u +delta))))
                if(f*f < eps):
#                    print("converged!")
                    break
        return q
    else:
        if u == 1:
            return 1
# Number of iterations(sample paths) for a given eta
iterations = 20


regret = np.full((iterations,(T.size - k+1)),0.0)

for iter in range(0,iterations):
    #Initializing mean estimate
    mean_est = np.full((k,2),0.0)

    max = np.full(k,0.0)
    i = 1
    for t in T:
        if(t < k):
            if (t == 0):
                mean_est[t] = (np.random.binomial(1,0.5,size = 1),1)
            else:
                mean_est[t] = (np.random.binomial(1,(0.5-(t+1)/70),size = 1),1)
        else:
            for j in range(0,k):
                max[j] = getupperbound(mean_est[j],t)
            arm = np.argmax(max)
            r = np.random.binomial(1,(0.5-(arm+1)/70),size = 1)
            mean_est[arm] = ((mean_est[arm,0]*mean_est[arm,1]+r)/(mean_est[arm,1]+1),(mean_est[arm,1]+1))
            regret[iter][i] = regret[iter][i-1] + np.random.binomial(1,0.5,size = 1)- r
            i = i+1
        print("Iteration & Time:", iter, t)


average_regret = np.mean(regret,axis = 0)
T = np.arange(0,average_regret.shape[0])
print(T)

plt.figure()
plt.plot( T, average_regret)
plt.show()
