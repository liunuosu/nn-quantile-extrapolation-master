from extreme.visualization import real_loglog_plot, real_hill_plot
from models import model_selection, get_best_crit, model_evaluation, model_selection_real, model_evaluation_real
from extreme.estimators import evt_estimators, evt_estimators_real
from extreme import visualization as simviz

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from extreme.data_management import DataLoader
from extension import christoffersen


#print(get_best_crit(filename="2023-06-08_12-28-17-rep1"))
#simviz.training_plot(k_anchor=100, show_as_video=False, epoch=50, saved=False,
                    #  NN="2023-06-07_15-21-21-rep1")
#print(model_selection_real(n_replications=1, distribution="empirical", params={'evi': 1.0, 'rho': [-0.27]}))
#simviz.xquantile_plot_real(NN="2023-06-08_12-28-17")
#simviz.xquantile_plot_real(NN="2023-06-08_12-58-51")
#simviz.xquantile_plot_real(NN="2023-06-08_13-37-20")
# print(get_best_crit(filename="2023-06-09_11-51-16-rep1"))
#print(model_selection_real(n_replications=1, distribution="empirical", params={'evi': 1.0, 'rho': [-1]}))
#print(model_evaluation_real(model_filename="2023-06-09_19-45-12", result_print=True))
#print(model_evaluation_real(model_filename="2023-06-09_19-35-14", result_print=True))
#print(model_evaluation_real(model_filename="2023-06-09_19-05-15", result_print=True))
#print(model_evaluation_real(model_filename="2023-06-09_19-17-58", result_print=True))

# 2023-06-09_19-45-12 NN2
# 2023-06-09_19-35-14 NN3
# 2023-06-09_19-05-15 NN4 for 2007-2009
# 2023-06-09_19-17-58 NN5
#simviz.xquantile_plot_real(NN="2023-06-09_19-05-15")
#print(evt_estimators_real(n_replications=1, distribution="empirical", n_data=1681, params={'evi': 1.0, 'rho': [-1]}))
#2023-06-08_13-59-45-rep1

#2012 data
# 2023-06-09_22-20-21 NN2
# 2023-06-09_22-25-43 NN3

#print(model_selection_real(n_replications=1, distribution="empirical", params={'evi': 1.0, 'rho': [-2]}))
#print(model_evaluation_real(model_filename="2023-06-09_22-20-21", result_print=True))
#print(model_evaluation_real(model_filename="2023-06-09_22-25-43", result_print=True))
#print(model_evaluation_real(model_filename="2023-06-09_22-34-04", result_print=True))
#print(model_evaluation_real(model_filename="2023-06-09_22-46-57", result_print=True))
#simviz.xquantile_plot_real(NN="2023-06-09_22-46-57")
#print(evt_estimators_real(n_replications=1, distribution="empirical", n_data=1681, params={'evi': 1.0, 'rho': [-2]}))

#print(model_selection_real(n_replications=1, distribution="empirical", params={'evi': 1.0, 'rho': [-3]}))
#print(model_evaluation_real(model_filename="2023-06-18_22-13-47", result_print=True))
#print(evt_estimators_real(n_replications=1, distribution="empirical", n_data=2514, params={'evi': 1.0, 'rho': [-3]}))

real_hill_plot()
plt.show()
#dataset = pd.read_csv("data_real/Volatility.csv")
#VIX = dataset["HS&P500"].values

#date = dataset["Date"].values

#interval = 502
#selected_dates = date[::interval]

#date_strings = [str(d) for d in selected_dates]
#plt.plot(VIX)
#plt.xticks(range(0, len(date), interval), date_strings)
#plt.xticks(rotation=30, ha='right')

# Giving x and y label to the graph
#plt.ylabel('Volatility')

#simviz.real_hill_plot()
#simviz.real_loglog_plot()
#simviz.real_hist_plot()

#print(model_selection_real(n_replications=1, distribution="empirical", params={'evi': 1.0, 'rho': [-2]}))
#print(model_evaluation_real(model_filename="2023-06-09_22-20-21", result_print=True))

#plt.show()