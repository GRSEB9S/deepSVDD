from datasets.base import DataLoader

from config import Configuration as Cfg

import numpy as np
import pandas as pd


class Adult_DataLoader(DataLoader):

    def __init__(self):

        DataLoader.__init__(self)

        self.dataset_name = "adult"

        self.n_train = 32561
        self.n_val = 16281
        self.n_test = 16281

        self.seed = Cfg.seed

        if Cfg.ad_experiment:
            self.n_classes = 2
        else:
            self.n_classes = 10

        Cfg.n_batches = int(np.ceil(self.n_train * 1. / Cfg.batch_size))

        self.data_path = "../data/adult.data"

        self.on_memory = True
        Cfg.store_on_gpu = True

        # load data from disk
        self.load_data()

    def load_data(self):

        print("Loading data...")

        names = ["age", "workclass", "fnlwgt", "education", "education-num",
                 "marital-status", "occupation", "relationship", "race", "sex",
                 "capital-gain", "capital-loss", "hours-per-week",
                 "native-country", "label"]

        # load data
        df = pd.read_csv(self.data_path, sep=',\s', header=None, names=names,
                         na_values=['?'], engine='python')

        # remove NAs
        df = df.dropna()

        # convert categorical variables


        # one-hot encode categorical features


        # extract X and y
        y = df.iloc[:, 0:-1]


        data_test = np.genfromtxt(self.data_path + 'adult.test')
