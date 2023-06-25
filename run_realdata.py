import datetime

import pandas as pd
from utils import get_config_real, save_config_summary_real
from extreme.data_management import DataLoader
from models_real import ExtrapolateNN_real, model_selection_real, model_evaluation_real
from pathlib import Path
import torch
import argparse
from extreme.estimators import evt_estimators_real

parser = argparse.ArgumentParser(description='Runner')
parser.add_argument('--processes', '-p',
                    help="number of processes. No multiprocessing by default",
                    default=1,
                    type=int)

args = parser.parse_args()
n_processes = args.processes

config = get_config_real()  # load .yaml configuration file
data_sampler = DataLoader()

# utilities
verbose = config["training"]["verbose"]
ckpt_epochs = [i for i in range(verbose, config["training"]["n_epochs"] + verbose, verbose)]  # check if all epochs have been trained
csv_pathfile = Path("ckpt_real", "_config_summary_real.csv")

now = datetime.datetime.now()
model_filename = now.strftime("%Y-%m-%d_%H-%M-%S")  # schema name model

def train_model():
    model_filename_rep = model_filename
    pathfile = Path("ckpt_real", "empirical", "training", "{}.pt".format(model_filename_rep))
    if pathfile.is_file():  # if the file exists, don't train it again !
        try:
            pt_ckpt = torch.load(pathfile, map_location="cpu")
            for chkpt_epoch in ckpt_epochs:  # check that all epochs have been trained
                pt_ckpt["epoch{}".format(chkpt_epoch)]["params"]
            return  # if it is the case, move to the next replication file
        except (EOFError, KeyError):  # if one is missing, remove the file and the csv line
            pathfile.unlink()  # file .pt
            df_summary = pd.read_csv(csv_pathfile, sep=";")   # associated csv row
            df_summary.drop(df_summary.index[(df_summary["model_filename"]==model_filename_rep)], inplace=True)
            df_summary.dropna(axis=0, how="all", inplace=True)
            df_summary.to_csv(csv_pathfile, header=True, index=False, sep=";")

    # regular training
    dataset = config["data"]["distribution"]

    data, X_order = data_sampler.load_real_data(dataset)
    total_data = data.shape[0]  # nb of log-spacings
    config["data"]["total_data"] = total_data

    order = config["model"]["trunc"]
    alpha = config["training"]["alpha"]

    save_config_summary_real(config=config, model_filename=model_filename_rep)  # save the config in a csv file

    print("="*10, "Training on {} dataset with order {} bias and alpha {} with a total of {} log spacings: {}".format(dataset, order, alpha,
                                                                                        total_data, model_filename_rep), "="*10)
    model = ExtrapolateNN_real(**config["model"], model_filename=model_filename_rep)
    model.train(data_train=data, X=X_order, distribution=dataset, **config["training"], real=True)
    print(model_evaluation_real(model_filename_rep, alpha, trunc=order, result_print=True))
    print(evt_estimators_real(distribution=dataset, alpha=alpha, trunc=order))
    return

if __name__ == '__main__':
    train_model()


