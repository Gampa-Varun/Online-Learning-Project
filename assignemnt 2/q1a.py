import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
import random
import bisect

#Number of Arms
k = 10
#Number of rounds
T = np.arange(0,25000)


# Number of iterations(sample paths) for a given eta
iterations = 20


regret = np.full((iterations,(T.size - k+1)),0.0)

for iter in range(0,iterations):
    #Initializing mean estimate
    mean_est = np.full((k,2),0.0)

    i = 1
    for t in T:
        if(t < k):
            if (t == 0):
                mean_est[t] = (np.random.binomial(1,0.5,size = 1),1)
            else:
                mean_est[t] = (np.random.binomial(1,(0.5-(t+1)/70),size = 1),1)
        else:
        #Choose to explore or exploit
            epsilon = 1/(t+1)
            expl = np.random.binomial(1,epsilon,size = 1)
            if (expl):
                arm = np.random.random_integers(0,k-1)
                if(arm>0):
                    r = np.random.binomial(1,(0.5-(arm+1)/70),size = 1)
                elif(arm == 0):
                    r = np.random.binomial(1,0.5,size = 1)
                mean_est[arm] = ((mean_est[arm,0]+r)/mean_est[arm,1],(mean_est[arm,1]+1))
            else:
                arm = np.argmax(mean_est,axis =0)[0]
                if(arm>0):
                    r = np.random.binomial(1,(0.5-(arm+1)/70),size = 1)
                elif(arm == 0):
                    r = np.random.binomial(1,0.5,size = 1)

            regret[iter][i] = regret[iter][i-1] + np.random.binomial(1,0.5,size = 1)- r
            i = i+1


average_regret = np.mean(regret,axis = 0)
T = np.arange(0,average_regret.shape[0])


freedom_degree = iterations - 1
error = ss.t.ppf(0.95, freedom_degree) * ss.sem(regret[:,-1])

plt.errorbar(T[-1], average_regret[-1], error,color='r' )

plt.plot( T, average_regret,color='r')
plt.show()
