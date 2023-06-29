
from models import model_selection, get_best_crit, model_evaluation, model_selection_real, model_evaluation_real
from extreme.estimators import evt_estimators, evt_estimators_real
from extreme import visualization as simviz
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from extreme.data_management import DataLoader
from extension import christoffersen, binomial_test

print(christoffersen("VXD", quantile=25.389, alpha=0.025))


p_value = binomial_test(actual_violations, expected_violations, confidence_level)
print("Binomial test p-value:", p_value)

