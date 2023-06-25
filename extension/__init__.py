import pandas as pd
import math
from scipy import stats


def christoffersen(column, quantile, alpha):
    dataset = pd.read_csv("data_real/Volatility_test.csv")
    data = dataset[column].values

    markov_chain = []

    for point in data:
        if point > quantile:
            obs = 1
        else:
            obs = 0
        markov_chain.append(obs)

    T1 = sum(markov_chain)
    T = len(markov_chain)
    T0 = T - T1
    cov_ratio = T1/T
    T00 = 0
    T01 = 0
    T10 = 0
    T11 = 0

    index = -1
    for point in markov_chain:
        if index == -1:  # if first one in list skip loop
            index += 1
            continue
        elif point == 1:
            if markov_chain[index] == 1:
                T11 += 1
            else:
                T01 += 1
        elif point == 0:
            if markov_chain[index] == 1:
                T10 += 1
            else:
                T00 += 1
        index += 1

    pi01_hat = T01/(T00 + T01)
    pi11_hat = T11/(T10 + T11)

    # COVERAGE LIKELIHOOD
    L_alpha = pow((1-alpha), T0) * pow(alpha, T1)
    L_pi = pow((1-cov_ratio), T0) * pow(cov_ratio, T1)
    L_uc = -2 * math.log(L_alpha/L_pi)

    # INDEPENDENCE LIKELIHOOD
    L_0 = pow((1-cov_ratio), T0) * pow(cov_ratio, T1)
    L_alt = pow((1-pi01_hat), T00) * pow(pi01_hat, T01) * pow((1-pi11_hat), T10) * pow(pi11_hat, T11)
    L_ind = -2 * math.log(L_0/L_alt)

    # CHRISTOFFERSEN TEST
    L_christoffersen = L_uc + L_ind
    p_value = 1 - stats.chi2.cdf(L_christoffersen, 2)

    return L_christoffersen, p_value
