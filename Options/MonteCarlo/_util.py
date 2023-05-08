import numpy as np
from numpy.random import binomial, uniform, normal, poisson

def Double_Expo_Rnd(n, p_up, eta1, eta2):
    bern = binomial(n=1, p=p_up, size=n)
    uni = uniform(low=0, high=1, size=n)

    dexpo = np.zeros(n)
    dexpo[bern == 1] = -np.log(uni[bern == 1]) / eta1
    dexpo[bern == 0] = np.log(uni[bern == 0]) / eta2
    return dexpo

def Mix_Norm_Rnd(n, p_up, a1, b1, a2, b2):
    bern = binomial(n=1, p=p_up, size=n)
    norm = normal(loc=0., scale=1., size=n)

    mnorm = np.zeros(n)
    mnorm[bern == 1] = a1 + b1 * norm[bern == 1]
    mnorm[bern == 0] = a2 + b2 * norm[bern == 0]
    return mnorm

