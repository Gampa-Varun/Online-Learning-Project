import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
import random
import bisect

#Number of Arms
arm_no = [10,20,30]

rounds = 25000

# Number of iterations(sample paths) for a given eta
iterations = 50

for k in arm_no:

    #Number of rounds
    T = np.arange(0,rounds)
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
        print( "No_arms and iterations of E-greedy iter- ", k, iter)


    average_regret = np.mean(regret,axis = 0)
    T = np.arange(0,average_regret.shape[0])

    plt.figure()
    freedom_degree = iterations - 1

    for bars in range(0,10):
        error = ss.t.ppf(0.95, freedom_degree) * ss.sem(regret[:,0 + bars*2500])
        plt.errorbar(T[0 +bars*2500], average_regret[0 + bars*2500], error, color = 'r')

    error = ss.t.ppf(0.95, freedom_degree) * ss.sem(regret[:,-1])
    plt.errorbar(T[-1], average_regret[-1], error, color = 'r')
    plt.plot( T, average_regret, color = 'r' , label = 'E-greedy')


    ##############################################################################

    #Number of rounds
    T = np.arange(0,rounds)
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

                upper_mean = np.take(mean_est,(0),axis = 1) + np.sqrt(1.5*np.log(t)/np.take(mean_est,(1),axis = 1))
                arm = np.argmax(upper_mean)
                r = np.random.binomial(1,(0.5-(arm+1)/70),size = 1)
                mean_est[arm] = ((mean_est[arm,0]*mean_est[arm,1]+r)/(mean_est[arm,1]+1),(mean_est[arm,1]+1))

                regret[iter][i] = regret[iter][i-1] + np.random.binomial(1,0.5,size = 1)- r
                i = i+1
        print( "No_arms and iterations of ucb- ",k, iter)

    average_regret = np.mean(regret,axis = 0)
    T = np.arange(0,average_regret.shape[0])

    for bars in range(0,10):
        error = ss.t.ppf(0.95, freedom_degree) * ss.sem(regret[:,0 + bars*2500])
        plt.errorbar(T[0 +bars*2500], average_regret[0 + bars*2500], error, color = 'b')

    error = ss.t.ppf(0.95, freedom_degree) * ss.sem(regret[:,-1])
    plt.errorbar(T[-1], average_regret[-1], error, color = 'b')
    plt.plot( T, average_regret, color = 'b' , label = 'UCB')






    ###############################################################################################################

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


    T = np.arange(0,rounds)
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

        print( "No_arms and iterations of kl-ucb- ",k, iter)

    average_regret = np.mean(regret,axis = 0)
    T = np.arange(0,average_regret.shape[0])

    for bars in range(0,10):
        error = ss.t.ppf(0.95, freedom_degree) * ss.sem(regret[:,0 + bars*2500])
        plt.errorbar(T[0 +bars*2500], average_regret[0 + bars*2500], error, color = 'g')


    error = ss.t.ppf(0.95, freedom_degree) * ss.sem(regret[:,-1])
    plt.errorbar(T[-1], average_regret[-1], error, color = 'g')
    plt.plot( T, average_regret, color = 'g' , label = 'kl-UCB')



    ############################################################################################################


    T = np.arange(0,rounds)

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

        print( "No_arms and iterations of Thompson_Sampling- ", k,iter)


    average_regret = np.mean(regret,axis = 0)
    T = np.arange(0,average_regret.shape[0])


    for bars in range(0,10):
        error = ss.t.ppf(0.95, freedom_degree) * ss.sem(regret[:,0 + bars*2500])
        plt.errorbar(T[0 +bars*2500], average_regret[0 + bars*2500], error, color = 'y')

    error = ss.t.ppf(0.95, freedom_degree) * ss.sem(regret[:,-1])
    plt.errorbar(T[-1], average_regret[-1], error, color = 'y')

    plt.plot( T, average_regret, color = 'y' , label = 'Thompson_Sampling')


    plt.legend(loc='upper center', bbox_to_anchor= ( 0.2, 0.6, 0.1, 0.4))
    plt.title("Cummulative Regret vs Time for T = 25000 and 50 Sample paths")
    plt.xlabel("Time")
    plt.ylabel("Cummulative Regret")

    #plt.show()
    if( k == 10):
        plt.savefig("Q1_10.png", bbox_inches='tight')
    if( k == 20):
        plt.savefig("Q1_20.png", bbox_inches='tight')
    if( k == 30):
        plt.savefig("Q1_30.png", bbox_inches='tight')
    plt.close()
