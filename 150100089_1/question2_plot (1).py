import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss


# Regret Data
# exp3_regret = [[R_1], ..., [R_i], ..., [R_N]]
# where [R_i] - [etaValue_i, regretSamplePath_i1, ..., regretSamplePath_iC]
# where 'N' is number of sample paths and 'C' is the total number of values that 'c' can take.
exp3_regret = [[], []]  # = [[...], ..., [etaValue_i, regretSamplePath_i1, ..., regretSamplePath_iC], ..., [...]]
exp3p_regret = [[], []]
exp3ix_regret = [[], []]


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
