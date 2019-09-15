import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
import random
import bisect

#Number of Arms
k = 10
#Number of rounds
T = np.arange(0,25000)


iterations = 20



regret = np.full((iterations,(T.size)),0.0)

for iter in range(0,iterations):

    Theta = np.full(k,0.0)
    S = np.full(k,0.0)
    F = np.full(k,0.0)


    for t in T:
        for i in range(0,k):
            Theta[i] = 1.0*np.random.beta((S[i]+1.0),(F[i] + 1.0))
        arm = np.argmax(Theta)
        max_ar_reward = np.random.binomial(1,0.5,size = 1) #reward of the optimal arm
        if(arm>0):
            r = np.random.binomial(1,(0.5-(arm+1)/70),size = 1)
        elif(arm == 0):
            r = max_ar_reward

        S[arm] = S[arm] + r
        F[arm] = F[arm] + 1-r
        if(t == 0):
            regret[iter][t] = max_ar_reward  - r
        else:
            regret[iter][t] = regret[iter][t-1] +max_ar_reward- r

        print("Time:", t)


average_regret = np.mean(regret,axis = 0)
T = np.arange(0,average_regret.shape[0])
print(T)

plt.figure()
plt.plot( T, average_regret)
plt.show()
