import pandas as pd

from models_real.network_real import ExtrapolateNN_real
from utils import load_summary_real, load_summary_file_real, nested_dict_to_df
from extreme.data_management import DataLoader
from extreme.estimators import random_forest_k
from models.metrics import compute_criteria
import torch
import numpy as np
from pathlib import Path
from extension import christoffersen


# we considered several criterias. For the paper, we selected the MAD
list_criterias = ["variance", "r_variance", "mad", "r_mad", "aad", "r_aad"]


def get_best_order_model(df_order_summary, criteria, condition):
    """returns the best model for a given order condition"""
    best_metric = None  # init the best metric
    for file in df_order_summary.iterrows():
        filename = file[1]["model_filename"]
        pt_ckpt = torch.load(Path("ckpt", condition, "{}.pt".format(filename)), map_location="cpu")
        metric = pt_ckpt["eval"][criteria]["value"]

        if best_metric is None or metric < best_metric:
            best_metric = metric
            best_epoch = pt_ckpt["eval"][criteria]["epoch"]
            best_filename = filename
    return best_filename, best_epoch, best_metric


def get_best_crit_real(filename, distribution=None):
    """returns the best epoch and value of a specific filename"""
    df = pd.DataFrame(columns=list_criterias, index=["epoch", "value"])
    if distribution is None:
        config_file = load_summary_file_real(filename=filename)
        distribution = config_file["distribution"]
    for criteria in list_criterias:
        pt_ckpt = torch.load(Path("ckpt_real", distribution, "training", "{}.pt".format(filename)), map_location="cpu")
        df.loc["epoch", criteria] = pt_ckpt["eval"][criteria]["epoch"]
        df.loc["value", criteria] = pt_ckpt["eval"][criteria]["value"]
    return df


def load_model_real(filename, epoch, distribution=None, from_config=False, config_file=None):
    # dict_models = {"sim": ExtrapolateNN, "LDreal": LDExtrapolateNN, "Creal": CExtrapolateNN}
    if not from_config:
        config_file = load_summary_file_real(filename=filename)
    model = ExtrapolateNN_real(**config_file)
    pt_ckpt = torch.load(Path("ckpt_real", distribution, "training", "{}.pt".format(filename)), map_location="cpu")
    model.net.load_state_dict(pt_ckpt["epoch{}".format(epoch)]["params"])
    model.optimizer.load_state_dict(pt_ckpt["epoch{}".format(epoch)]["optimizer"])

    return model


def model_selection_real(distribution, trunc, alpha, metric="median", **kwargs):
    """
    returns the best models for each NN order condition and eval criteria, given a specific parametrization
    Parameters
    ----------
    distribution : str
    metric : str
    kwargs :

    Returns
    -------

    """
    pathdir = Path("ckpt_real", distribution, "extrapolation", str(alpha) + str("_") + str(trunc))
    pathdir.mkdir(parents=True, exist_ok=True)

    df_summary = load_summary_real()
    df_summary = df_summary[df_summary["distribution"] == distribution]  # filter on the distributions
    df_summary = df_summary[df_summary["trunc"] == trunc]

    try:
        nn_order_trunc = np.sort(df_summary["trunc"].unique())  # list the different neural network truncation
    except KeyError:
        print("No such parametrization is listed !")

    dict_best_nn = {
        "NN_{}".format(order): {crit: {"filename": None, "value": None, "rmse_bestK": None} for crit in list_criterias}
        for order in
        nn_order_trunc}  # save the best extrapolation model for each criteria and orderNN

    for order in nn_order_trunc:  # for each NN order condition
        df_order_summary = df_summary[(df_summary["trunc"] == order)]
        model_filename = df_order_summary["model_filename"]

        dict_nn = model_evaluation_real(model_filename, alpha, trunc)
        for criteria in list_criterias:
            # condition to find the best model given each NN order and criteria (just windowed)
            best_metric = dict_best_nn["NN_{}".format(order)][criteria]["value"]
            if best_metric is None or dict_nn[criteria][metric]["value"] < best_metric:
                dict_best_nn["NN_{}".format(order)][criteria]["value"] = dict_nn[criteria][metric]["value"]
                dict_best_nn["NN_{}".format(order)][criteria]["filename"] = model_filename
                dict_best_nn["NN_{}".format(order)][criteria]["rmse_bestK"] = dict_nn[criteria][metric][
                    "rmse_bestK"]

    return nested_dict_to_df(dict_best_nn).T


def model_evaluation_real(model_filename, alpha, trunc, result_print=False):
    df_summary = load_summary_real()
    config_model = df_summary[df_summary["model_filename"] == model_filename]

    print("running {} ...".format(model_filename))

    # initialize utilities
    # Change
    distribution = config_model["distribution"].values[0]
    pathdir = Path("ckpt_real", distribution, "extrapolation", str(alpha) + str("_") + str(trunc))
    pathdir.mkdir(parents=True, exist_ok=True)
    pathfile = Path(pathdir, "{}.npy".format(model_filename))
    try:
        dict_nn = np.load(pathfile, allow_pickle=True)[()]
    except FileNotFoundError:
        # save all info in a new dictionary
        dict_nn = {criteria: {_metric: {"value": [], "series": [],
                                        "var": None, "rmse": None, "rmse_bestK": None, "q_bestK": [],
                                        "bestK": []} for _metric in ["mean", "median"]} for criteria in
                    list_criterias}

        data_sampler = DataLoader()
        data, X_order = data_sampler.load_real_data(distribution)
        anchor_points = np.arange(2, len(X_order))  # k = 2, ..., n-1
        EXTREME_ALPHA = alpha

        # best epoch and value at each replication
        best_parametrization = get_best_crit_real(filename=model_filename, distribution=distribution)
        for criteria in list_criterias:  # for each criteria

            best_epoch_rep = int(best_parametrization.loc["epoch", criteria])

            # NN extrapolation
            model = load_model_real(model_filename, best_epoch_rep, distribution)
            # extrapolation
            q_nn = model.extrapolate(alpha=EXTREME_ALPHA, k_anchor=anchor_points, X_order=X_order).ravel()

            # find the best k
            bestK_nn = random_forest_k(q_nn, 10000)  # for k=15,...,375 (i=13,...,373)

            # MEAN (RMSE)
            dict_nn[criteria]["mean"]["value"].append(best_parametrization.loc["value", criteria])
            dict_nn[criteria]["mean"]["series"].append(q_nn)
            dict_nn[criteria]["mean"]["q_bestK"].append(q_nn[int(bestK_nn)])
            dict_nn[criteria]["mean"]["bestK"].append(bestK_nn + 2)  # k = i +2, with Python index i=(0,...,497)

            # MEDIAN (RMedSE)
            dict_nn[criteria]["median"]["value"].append(best_parametrization.loc["value", criteria])
            dict_nn[criteria]["median"]["series"].append(q_nn)
            dict_nn[criteria]["median"]["q_bestK"].append(q_nn[int(bestK_nn)])
            dict_nn[criteria]["median"]["bestK"].append(
                bestK_nn + 2)  # k = i +2, with Python index i=(0,...,497)

        for criteria in list_criterias:
            # MEAN (RMSE)
            q_nn_mean_series = np.array(dict_nn[criteria]["mean"]["series"])
            dict_nn[criteria]["mean"]["value"] = np.mean(
                dict_nn[criteria]["mean"]["value"])  # mean of all best values in order to compare NN models
            dict_nn[criteria]["mean"]["var"] = q_nn_mean_series.var(axis=0)  # variance between the replications
            dict_nn[criteria]["mean"]["series"] = q_nn_mean_series.mean(axis=0)  # mean between the replications

            # MEDIAN (RMedSE)
            q_nn_median_series = np.array(dict_nn[criteria]["median"]["series"])
            dict_nn[criteria]["median"]["value"] = np.median(
                dict_nn[criteria]["median"]["value"])  # median of all best values in order to compare NN models
            dict_nn[criteria]["median"]["var"] = q_nn_median_series.var(axis=0)  # variance between the replications
            dict_nn[criteria]["median"]["series"] = np.median(q_nn_median_series,
                                                                axis=0)  # median between the replications
        np.save(pathfile, dict_nn)
    finally:
        if not result_print:
            return dict_nn
        else:
            df = pd.DataFrame(columns=list_criterias, index=["K", "quantile"])
            metric = "median"
            for criteria in list_criterias:
                df.loc["K", criteria] = dict_nn[criteria][metric]["bestK"][0]
                df.loc["quantile", criteria] = np.round(np.array(dict_nn[criteria][metric]["q_bestK"]).ravel()[0],
                                                        4)
                # ADDED
            return df




