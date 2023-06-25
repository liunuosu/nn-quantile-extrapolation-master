import torch
import numpy as np
import pandas as pd

from models import model_selection, get_best_crit, model_evaluation
from extreme.estimators import evt_estimators
from extreme import visualization as simviz

import matplotlib.pyplot as plt

#print(get_best_crit(filename="2022-05-03_09-55-01-rep5"))
#print(get_best_crit(filename="2023-05-23_13-44-34-rep1"))
#print(get_best_crit(filename="2023-05-24_16-40-59-rep1"))
#simviz.training_plot(k_anchor=100, show_as_video=False, epoch=488, saved=False,
         #             NN="2023-05-24_16-40-59-rep1")
#plt.show()

#print(get_best_crit(filename="2023-05-24_17-08-55-rep1"))
#simviz.training_plot(k_anchor=100, show_as_video=False, epoch=418, saved=False,
 #                     NN="2023-05-24_17-08-55-rep1")
#plt.show()

# ALL WITH ORDER CONDITION 4
#GPD distribution with gamma = 0.125
#print(model_selection(n_replications=10, distribution="gpd", params={"evi":0.125, "rho":[-0.25]}))
#simviz.xquantile_plot(NN="2023-05-25_18-40-10")

#NHW distribution with gamma = 1 and rho = -1/8
#print(model_selection(n_replications=10, distribution="nhw", params={"evi":1., "rho":[-0.125]}))
#simviz.xquantile_plot(NN="2023-05-25_20-16-07")

# BURR distribution with gamma = 1 and rho = -2
#print(model_selection(n_replications=10, distribution="burr", params={"evi":1., "rho":[-2]}))
#simviz.xquantile_plot(NN="2023-05-25_21-45-21")

#print(model_selection(n_replications=10, distribution="burr", params={"evi":1., "rho":[-0.25]}))
#simviz.xquantile_plot(NN="2023-05-24_17-08-55")

#print(model_selection(n_replications=10, distribution="fisher", params={"evi":0.125, "rho":[-0.25]}))
#simviz.xquantile_plot(NN="2023-05-26_16-34-08")

print(model_selection(n_replications=1, distribution="fisher", params={"evi":1.0, "rho":[-0.25]}))
simviz.xquantile_plot(NN="2023-06-07_17-33-28")

# CONSIDER GPD Distribution with other order conditions (order = 2)
# Training on GPD distribution ({'evi': 0.125, 'rho': [-2]}) with a total of 124251 data: 2023-05-25_23-36-05
#print(model_selection(n_replications=10, distribution="gpd", params={"evi":0.125, "rho":[-0.25]}))
#simviz.xquantile_plot(NN="2023-05-26_10-40-56")

# GPD ORDER = 3 ({'evi': 0.125, 'rho': [-0.25]}) with a total of 124251 data: 2023-05-26_00-57-23
#print(model_selection(n_replications=10, distribution="gpd", params={"evi":0.125, "rho":[-0.25]}))
#simviz.xquantile_plot(NN="2023-05-25_18-40-10")

# GPD ORDER = 5
#print(model_selection(n_replications=10, distribution="gpd", params={"evi":0.125, "rho":[-0.25]}))
#J = 2
#simviz.xquantile_plot(NN="2023-05-26_11-40-19")
#J = 3
#simviz.xquantile_plot(NN="2023-05-26_00-57-23")
#J = 4
#simviz.xquantile_plot(NN="2023-05-25_18-40-10")
#J = 5
#simviz.xquantile_plot(NN="2023-05-26_01-56-59")
# RE RUN FOR order = 2

#Training on STUDENT distribution ({'evi': 1.0, 'rho': [-0.25]}) with a total of 124251 data: 2023-05-26_18-20-55-rep10
#print(model_selection(n_replications=10, distribution="student", params={"evi":1.0, "rho":[-0.25]}))
#simviz.xquantile_plot(NN="2023-05-26_18-20-55")

#INVGAMMA distribution ({'evi': 1.0, 'rho': [-0.25]}) with a total of 124251 data: 2023-05-26_20-08-16-rep10
#print(model_selection(n_replications=10, distribution="invgamma", params={"evi":1.0, "rho":[-0.25]}))
#simviz.xquantile_plot(NN="2023-05-26_20-08-16")

#Training on FISHER distribution ({'evi': 1.0, 'rho': [-0.25]}) with a total of 124251 data: 2023-05-26_21-45-57-rep10
#print(model_selection(n_replications=10, distribution="fisher", params={"evi":1.0, "rho":[-0.25]}))
#simviz.xquantile_plot(NN="2023-05-26_21-45-57")

#print(model_selection(n_replications=10, distribution="gpd", params={"evi":0.125, "rho":[-0.25]}))
#simviz.xquantile_plot(NN="2023-05-26_01-56-59")

#print(model_selection(n_replications=10, distribution="burr", params={"evi":1., "rho":[-0.25]}))
#simviz.xquantile_plot(NN="2023-05-24_17-08-55")

#print(model_selection(n_replications=10, distribution="gpd", params={"evi":0.125, "rho":[-0.25]}))
#simviz.xquantile_plot(NN="2023-05-26_11-40-19")



#print(get_best_crit(filename="2023-05-24_17-08-55-rep2"))
#simviz.training_plot(k_anchor=100, show_as_video=False, epoch=463, saved=False,
#                      NN="2023-05-24_17-08-55-rep2")
plt.show()


# get_best_crit(filename="2023-05-23_12-28-14-rep1")
#simviz.training_plot(k_anchor=100, show_as_video=False, epoch=495, saved=False,
              #        NN="2023-05-23_13-44-34-rep1")
#simviz.training_plot(k_anchor=100, show_as_video=False, epoch=495, saved=False,
               #       NN="2023-05-23_13-44-34-rep1")


#simviz.training_plot(k_anchor=100, show_as_video=False, epoch=1, saved=False,
 #                     NN="2023-05-23_12-28-14-rep1")
#model_selection(distribution="burr", params={"evi":1., "rho":[-0.25]}, n_replications=1)
# model_selection(distribution="burr", params={"evi":1., "rho":[-0.125]}, n_replications=500)
#simviz.xquantile_plot(NN="2023-05-23_13-44-34")
#plt.show()