
from models import model_selection, get_best_crit, model_evaluation, model_selection_real, model_evaluation_real
from extreme.estimators import evt_estimators, evt_estimators_real
from extreme import visualization as simviz
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from extreme.data_management import DataLoader
from extension import christoffersen

print(christoffersen("VIX", 32, 0.025))
