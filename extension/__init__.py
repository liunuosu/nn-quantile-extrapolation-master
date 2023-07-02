import pandas as pd
import math
from scipy import stats
from scipy.stats import binom_test
import numpy as np
from arch import arch_model


def GARCH():
    # Set the seed for reproducibility
    np.random.seed(0)

    # Define the GARCH parameters
    omega = 0.01
    alpha = 0.1
    beta = 0.8

    # Define the number of observations to simulate
    n_obs = 1002

    # Create empty arrays to store the simulated returns and volatility
    returns = np.zeros(n_obs)
    volatility = np.zeros(n_obs)

    # Simulate the GARCH process
    garch_model = arch_model(None, vol='GARCH', p=1, q=1)
    for i in range(n_obs):
        if i == 0:
            returns[i] = np.random.normal(0, 1)  # Initial return
            volatility[i] = np.sqrt(omega / (1 - alpha - beta))  # Initial volatility
        else:
            conditional_volatility = np.sqrt(omega + alpha * returns[i - 1] ** 2 + beta * volatility[i - 1] ** 2)
            returns[i] = conditional_volatility * np.random.normal(0, 1)
            volatility[i] = conditional_volatility

    # Print the simulated returns and volatility


    return returns, volatility

def christoffersen(column, quantile, alpha):
    dataset = pd.read_csv("data_real/Volatility_test.csv")
    data = dataset[column].values

    #returns, volatility = GARCH()
    #data = volatility[501:1001]

    markov_chain = []

    for point in data:
        if point > quantile:
            obs = 1
        else:
            obs = 0
        markov_chain.append(obs)

    T1 = sum(markov_chain)
    print("T1: " + str(T1))
    T = len(markov_chain)
    print("T: " + str(T))
    T0 = T - T1
    cov_ratio = T1/T
    print("Coverage: " + str(cov_ratio))
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

    print("T00: " + str(T00))
    print("T01: " + str(T01))
    print("T10: " + str(T10))
    print("T11: " + str(T11))

    pi01_hat = T01/(T00 + T01)
    pi11_hat = T11/(T10 + T11)

    #print("pi01_hat " + str(pi01_hat))
    #print("pi11_hat " + str(pi11_hat))

    # COVERAGE LIKELIHOOD
    L_alpha = pow((1-alpha), T0) * pow(alpha, T1)
    print("L_alpha: " + str(L_alpha))
    L_pi = pow((1-cov_ratio), T0) * pow(cov_ratio, T1)
    print("L_pi: "+ str(L_pi))
    L_uc = -2 * math.log(L_alpha/L_pi)
    print("L_cov: " + str(L_uc))

    # INDEPENDENCE LIKELIHOOD
    L_0 = pow((1-cov_ratio), T0) * pow(cov_ratio, T1)
    L_alt = pow((1-pi01_hat), T00) * pow(pi01_hat, T01) * pow((1-pi11_hat), T10) * pow(pi11_hat, T11)
    L_ind = -2 * math.log(L_0/L_alt)
    print("L_ind: " + str(L_ind))

    # CHRISTOFFERSEN TEST

    p_value_kupiec = 1 - stats.chi2.cdf(L_uc, 1)
    print("Kupiec Likelihood: " + str(L_uc))
    print("P-value kupiec: " + str(p_value_kupiec))

    L_christoffersen = L_uc + L_ind
    p_value = 1 - stats.chi2.cdf(L_christoffersen, 2)

    print("Christoffersen likelihood: " + str(L_christoffersen))
    print("P-value christoffersen: " + str(p_value))

    return L_christoffersen, p_value
    #return 0, 0


def binomial_test(actual_violations, expected_violations, confidence_level):
    """
    Perform the binomial test to evaluate VaR violations.

    Parameters:
    actual_violations (int): Number of actual VaR violations.
    expected_violations (int): Expected number of VaR violations under the VaR model.
    confidence_level (float): Desired confidence level (e.g., 0.95 for 95% confidence).

    Returns:
    p_value (float): The p-value of the binomial test.

    """
    p_value = binom_test(actual_violations, n=actual_violations + expected_violations, p=confidence_level)

    return p_value

