import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
import random
import bisect


d = 10
T = np.power(10,5)

def reward_generator(delta):
    p = 0.5 + delta
    return(1-np.random.binomial(1,p,size = 1))

def choose (weights,d):
    cdf = np.empty(d)
    cummulative = 0

    k = 0

    for w in weights:
        cummulative += w
        cdf[k] = cummulative
        k = k + 1

    x = random.random()
    idx = np.searchsorted(cdf, x)


    return idx


for c in range(1,22,2):
    eta = 0.1*c*0.95*np.sqrt(np.log(d)/(d*T))
    regret = 0
    gamma = 1.05*np.sqrt(d*np.log(d)/T)
    beta = np.sqrt(np.log(d)/(d*T))

    for iter in range(0,50):

        Total_pl_loss = 0
        S = np.full(d,0)
        Total_loss = np.full(d,0.0)
        player_loss = np.full(d,0)


        delta = 0.1

        #Initializing loss
        loss = np.empty(d)
        P = np.exp(eta*S)
        P = P/np.sum(P)

        for i in range(0,T):



            #Choosing an expert
            expert = choose(P,d)


            #Computing loss
            for j in range(0,10):
                if j < 8:
                    loss[j] = reward_generator(0)
                elif j == 8:
                    loss[j] = reward_generator(-1*delta)
                elif j == 9:
                    if i < T/2:
                        loss[j] = reward_generator(delta)
                    else:
                        loss[j] = reward_generator(-2*delta)

            estimated_gains = beta/P
            estimated_gains[expert] += loss[expert]

            S = S + estimated_gains

            P = (1-gamma)*np.exp(eta*S)/np.sum(np.exp(eta*S)) + gamma/d

            if expert < 8:
                Pl_loss = 0.5
            elif expert == 8:
                Pl_loss = 0.5 + delta
            elif expert == 9:
                if i < T/2:
                    Pl_loss = 0.5 - delta
                else:
                    Pl_loss = 0.5 + 2*delta


            Total_pl_loss += Pl_loss
            Total_loss += loss



        max_loss = np.amax(Total_loss)
        regret += max_loss - Total_pl_loss


    print(regret/50)
    #print(2*np.sqrt(T*d*np.log(d)))
