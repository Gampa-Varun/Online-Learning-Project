import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
import random
import bisect


# Regret Data
# exp3_regret = [[R_1], ..., [R_i], ..., [R_N]]
# where [R_i] - [etaValue_i, regretSamplePath_i1, ..., regretSamplePath_iC]
# where 'N' is number of sample paths and 'C' is the total number of values that 'c' can take.
exp3_regret = []  # = [[...], ..., [etaValue_i, regretSamplePath_i1, ..., regretSamplePath_iC], ..., [...]]
exp3p_regret = []
exp3ix_regret = []



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


iterations = 5
count = 0

for c in range(1,22,2):
    eta = 0.1*c*np.sqrt(np.log(d)/(d*T))
    reg = np.full((11,iterations),0.0)


    for iter in range(0,iterations):

        Total_pl_loss = 0.0
        S = np.full(d,0.0)
        Total_loss = np.full(d,0.0)


        delta = 0.1

        #Initializing loss
        loss = np.empty(d)

        for i in range(0,T):
            P = np.exp(eta*S)

            P = P/np.sum(P)

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

            estimated_gain = loss[expert]/P[expert]
            S[expert] = S[expert] + estimated_gain

            if expert < 8:
                Pl_loss = 0.5
            elif expert == 8:
                Pl_loss = 0.5 + delta
            elif expert == 9:
                if i < T/2:
                    Pl_loss = 0.5 - delta
                else:
                    Pl_loss = 0.5 + 2*delta

#Expected values of rewards to get expected Pseudo regret
            Total_pl_loss += Pl_loss
            Total_loss += loss

        max_loss = np.amax(Total_loss)
        reg[count][iter] =  (max_loss - Total_pl_loss)

    count = count+1

    exp3_regret.append([eta]+reg[count-1].tolist())


#Exp3P
count = 0
for c in range(1,22,2):
    eta_p = 0.1*c*0.95*np.sqrt(np.log(d)/(d*T))
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
#Expected values of rewards to get expected Pseudo regret
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
        reg_p[count][iter] =( max_loss - Total_pl_loss)

    count = count+1

    exp3p_regret.append([eta_p]+reg_p[count-1].tolist())


#Exp3ix
count = 0
for c in range(1,22,2):

    eta = 0.1*c*0.95*np.sqrt(np.log(d)/(d*T))
    reg_ix =  np.full((11,iterations),0.0)
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

        #Initializing distribution
        P = np.exp(eta*S)
        P = P/np.sum(P)

        for i in range(0,T):

            P = np.exp(eta*S)
            P = P/np.sum(P)

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

            estimated_gain = loss[expert]/(P[expert]+gamma)

            S[expert] = S[expert] + estimated_gain




            P = (1-gamma)*np.exp(eta_p*S)/np.sum(np.exp(eta_p*S)) + gamma/d

#Expected values of rewards to get expected Pseudo regret
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
        reg_ix[count][iter] = (max_loss - Total_pl_loss)

    count = count+1

    exp3ix_regret.append([eta]+reg_ix[count-1].tolist())



# Plotting Regret vs Eta
# EXP3
eta = []
regret_mean = []
regret_err = []
freedom_degree = len(exp3_regret[0]) - 2
for regret in exp3_regret:
    eta.append(regret[0])
    regret_mean.append(np.mean(regret[1:]))
    regret_err.append(ss.t.ppf(0.95, freedom_degree) * ss.sem(regret[1:]))


colors = list("rgbcmyk")
shape = ['--^', '--d', '--v']
plt.errorbar(eta, regret_mean, regret_err, color=colors[0])
plt.plot(eta, regret_mean, colors[0] + shape[0], label='EXP3')

# EXP3.P
eta = []
regret_mean = []
regret_err = []
freedom_degree = len(exp3p_regret[0]) - 2

for regret in exp3p_regret:
    eta.append(regret[0])
    regret_mean.append(np.mean(regret[1:]))
    regret_err.append(ss.t.ppf(0.95, freedom_degree) * ss.sem(regret[1:]))


plt.errorbar(eta, regret_mean, regret_err, color=colors[1])
plt.plot(eta, regret_mean, colors[1] + shape[1], label='EXP3.P')


# EXP3-IX
eta = []
regret_mean = []
regret_err = []
freedom_degree = len(exp3ix_regret[0]) - 2
for regret in exp3ix_regret:
    eta.append(regret[0])
    regret_mean.append(np.mean(regret[1:]))
    regret_err.append(ss.t.ppf(0.95, freedom_degree) * ss.sem(regret[1:]))


plt.errorbar(eta, regret_mean, regret_err, color=colors[2])
plt.plot(eta, regret_mean, colors[2] + shape[2], label='EXP3-IX')


# Plotting
plt.legend(loc='upper right', numpoints=1)
plt.title("Pseudo Regret vs Learning Rate for T = 10^5 and 20 Sample paths")
plt.xlabel("Learning Rate")
plt.ylabel("Pseudo Regret")
plt.savefig("Q2.png", bbox_inches='tight')
plt.close()
