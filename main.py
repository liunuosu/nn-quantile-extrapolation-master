
from models import model_selection, get_best_crit, model_evaluation, model_selection_real, model_evaluation_real
from extreme.estimators import evt_estimators, evt_estimators_real
from extreme import visualization as simviz
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from extreme.data_management import DataLoader
from extension import christoffersen, binomial_test, GARCH

# Run the training results by running run_realdata.py
# The parameters can be adjusted in config_file_real.yaml
# Results are stored in ckpt_real

# Run the test here, kupiec test results also printed
# When you want to run the GARCH, uncomment the two comments in extreme.data_management
# and uncomment the two comments in extension.init

# First is the distribution, second input is the quantile, and third input the alpha
christoffersen("VIX", 0.421, 0.05)

