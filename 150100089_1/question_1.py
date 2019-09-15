import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
import random
import bisect

#Number of Experts
d = 10
#Number of rounds
T = np.power(10,3)

#Regret
wm_regret = []

def loss_generator(delta):
    p = 0.5 + delta
    return(np.random.binomial(1,p,size = 1))

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


# Number of iterations(sample paths) for a given eta
iterations = 20

count = 0


for c in range (1,22,2):

    reg_wm = np.full((11,iterations),0.0)
    #parameter eta, delta
    eta = 0.1*c*np.sqrt(2*np.log(d)/T)
    delta = 0.1
    print(c)

    for iter in range(0,iterations):

        #Initializing weights
        W_ =  np.full(d,1)

        #Parameters to claculate regret
        Total_loss = np.empty(d)
        iel = 0
        print("check")
        #Initializing loss
        loss = np.empty(d)

        for i in range(0,T):
                W = W_/np.sum(W_)

                #Choosing an expert
                expert = choose(W,d)

                #Computing loss
                for j in range(0,10):
                    if j < 8:
                        loss[j] = loss_generator(0)
                    elif j == 8:
                        loss[j] = loss_generator(-1*delta)
                    elif j == 9:
                        if i < T/2:
                            loss[j] = loss_generator(delta)
                        else:
                            loss[j] = loss_generator(-2*delta)

                player_loss = loss[expert]
                #print(player_loss)

                #inner expected loss
                iel += np.dot(W,loss)

                Total_loss += loss
                #Update
                W_ = np.multiply(W_,np.exp(-eta*loss))

        minimum_loss = np.amin(Total_loss)


        reg_wm[count][iter] = iel - minimum_loss

    count = count + 1
    wm_regret.append([eta]+reg_wm[count-1].tolist())



# Collecting
eta = []
regret_mean = []
regret_err = []
freedom_degree = len(wm_regret[0]) - 2
for regret in wm_regret:
    eta.append(regret[0])
    regret_mean.append(np.mean(regret[1:]))
    regret_err.append(ss.t.ppf(0.95, freedom_degree) * ss.sem(regret[1:]))

colors = list("rgbcmyk")
shape = ['--^', '--d', '--v']
plt.errorbar(eta, regret_mean, regret_err, color=colors[2])
plt.plot(eta, regret_mean, colors[2] + shape[2], label='W-M')


# Plotting
plt.legend(loc='upper right', numpoints=1)
plt.title("Pseudo Regret vs Learning Rate for T = 10^5 and 20 Sample paths")
plt.xlabel("Learning Rate")
plt.ylabel("Pseudo Regret")
plt.savefig("Q1.png", bbox_inches='tight')
plt.close()
    #print("bound",np.sqrt(2*T*np.log(d)))
