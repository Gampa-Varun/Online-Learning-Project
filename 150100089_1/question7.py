import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
import random
import bisect


d = 2
T = np.power(10,3)
iterations = 20
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

exp3p_regret = []
#Exp3P
count = 0
for c in range(1,22,2):
    eta = 0.1*c*0.95*np.sqrt(np.log(d)/(d*T))
    reg_p = np.full((11,iterations),0.0)
    gamma = 1.05*np.sqrt(d*np.log(d)/T)
    beta = np.sqrt(np.log(d)/(d*T))

    for iter in range(0,iterations):

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
            for j in range(0,d):
                if j < 1:
                    loss[j] = reward_generator(0)

                else:
                    loss[j] = reward_generator(0.05)

            estimated_gains = beta/P
            estimated_gains[expert] += loss[expert]

            S = S + estimated_gains

            P = (1-gamma)*np.exp(eta*S)/np.sum(np.exp(eta*S)) + gamma/d
#Expected values of rewards to get expected Pseudo regret
            if expert < 1:
                Pl_loss = 0.5

            else:
                Pl_loss = 0.55


            Total_pl_loss += Pl_loss
            Total_loss += loss


        print("Check")
        max_loss = np.amax(Total_loss)
        reg_p[count][iter] = max_loss - Total_pl_loss

    count = count+1

    exp3p_regret.append([eta]+reg_p[count-1].tolist())


# EXP3.P
eta = []
regret_mean = []
regret_err = []
freedom_degree = len(exp3p_regret[0]) - 2

for regret in exp3p_regret:
    eta.append(regret[0])
    regret_mean.append(np.mean(regret[1:]))
    regret_err.append(ss.t.ppf(0.95, freedom_degree) * ss.sem(regret[1:]))


colors = list("rgbcmyk")
shape = ['--^', '--d', '--v']
plt.errorbar(eta, regret_mean, regret_err, color=colors[1])
plt.plot(eta, regret_mean, colors[1] + shape[1], label='EXP3.P')

# Plotting
plt.legend(loc='upper right', numpoints=1)
plt.title("Pseudo Regret vs Learning Rate for T = 10^5 and 20 Sample paths")
plt.xlabel("Learning Rate")
plt.ylabel("Pseudo Regret")
plt.savefig("Q7.png", bbox_inches='tight')
plt.close()
