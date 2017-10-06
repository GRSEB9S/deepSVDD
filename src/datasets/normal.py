import numpy as np
import lasagne.nonlinearities
import lasagne.layers

from base import DataLoader
from utils.misc import flush_last_line
from config import Configuration as Cfg


class Normal_DataLoader(DataLoader):

    def __init__(self):

        DataLoader.__init__(self)

        self.dataset_name = "normal"
        self.data_path = ""

        self.n_train = Cfg.toy_n_train
        self.n_val = int(np.ceil(0.2 * Cfg.toy_n_train))
        self.n_test = int(np.ceil(0.2 * Cfg.toy_n_train))

        self.n_classes = 2  # normal and anomalous/outlier

        Cfg.n_batches = int(np.ceil(self.n_train * 1. / Cfg.batch_size))

        self.on_memory = True
        Cfg.store_on_gpu = True

        # generation parameters
        self.seed = Cfg.seed
        self.dim = Cfg.toy_ndim

        # generate data
        self.load_data()

        # add anomalous data
        self.add_outliers(frac=Cfg.out_frac)

        # shuffle data (since batches are extracted block-wise)
        perm_train = np.random.permutation(len(self._y_train))
        self._X_train = self._X_train[perm_train]
        self._y_train = self._y_train[perm_train]

        perm_val = np.random.permutation(len(self._y_val))
        self._X_val = self._X_val[perm_val]
        self._y_val = self._y_val[perm_val]

        perm_test = np.random.permutation(len(self._y_test))
        self._X_test = self._X_val[perm_test]
        self._y_test = self._y_val[perm_test]

        # Set data type for use with theano
        self._X_train = self._X_train.astype(np.float32)
        self._X_val = self._X_val.astype(np.float32)
        self._X_test = self._X_test.astype(np.float32)
        self._y_train = self._y_train.astype(np.uint8)
        self._y_val = self._y_val.astype(np.uint8)
        self._y_test = self._y_test.astype(np.uint8)

    def load_data(self):
        """
        Generate data from the standard normal
        """

        print("Generating data...")

        # generate data
        np.random.seed(self.seed)

        self._X_train = np.random.normal(10, 1, (self.n_train, self.dim))
        # self._X_train = np.random.normal(0, 1, (self.n_train, self.dim))
        self._y_train = np.zeros(self.n_train)

        self._X_val = np.random.normal(10, 1, (self.n_val, self.dim))
        # self._X_val = np.random.normal(0, 1, (self.n_val, self.dim))
        self._y_val = np.zeros(self.n_val)

        self._X_test = np.random.normal(10, 1, (self.n_test, self.dim))
        # self._X_test = np.random.normal(0, 1, (self.n_test, self.dim))
        self._y_test = np.zeros(self.n_test)

        flush_last_line()
        print("Data generated.")

    def add_outliers(self, frac=0.1):
        """
        For outliers, noise from N(0,10) is added to one randomly chosen dim.
        """

        idx_out_train = np.arange(int(np.ceil(self.n_train * (1-frac))),
                                  self.n_train, 1)
        idx_out_val = np.arange(int(np.ceil(self.n_val * (1 - frac))),
                                self.n_val, 1)
        idx_out_test = np.arange(int(np.ceil(self.n_test * (1 - frac))),
                                 self.n_test, 1)

        n_out_train = self.n_train - int(np.ceil(self.n_train * (1-frac)))
        n_out_val = self.n_val - int(np.ceil(self.n_val * (1 - frac)))
        n_out_test = self.n_test - int(np.ceil(self.n_test * (1 - frac)))

        dim_out_train = np.random.choice(np.arange(self.dim), n_out_train)
        dim_out_val = np.random.choice(np.arange(self.dim), n_out_val)
        dim_out_test = np.random.choice(np.arange(self.dim), n_out_test)

        # add noise
        self._X_train[idx_out_train, dim_out_train] = np.random.normal(
            1, 0.1, n_out_train)
        self._X_val[idx_out_val, dim_out_val] = np.random.normal(
            1, 0.1, n_out_val)
        self._X_test[idx_out_test, dim_out_test] = np.random.normal(
            1, 0.1, n_out_test)

        # self._X_train[idx_out_train, dim_out_train] += np.random.normal(
        #     0, 10, n_out_train)
        # self._X_val[idx_out_val, dim_out_val] += np.random.normal(
        #     0, 10, n_out_val)
        # self._X_test[idx_out_test, dim_out_test] += np.random.normal(
        #     0, 10, n_out_test)

        self._y_train[idx_out_train] = 1
        self._y_val[idx_out_val] = 1
        self._y_test[idx_out_test] = 1

    def build_architecture(self, nnet):

        if Cfg.dropout_architecture:
            units_mult = 2
        else:
            units_mult = 1

        nnet.addInputLayer(shape=(None, self.dim))

        if Cfg.dropout:
            nnet.addDropoutLayer(p=0.2)

        for layer in range(Cfg.toy_net_depth):
            nnet.addDenseLayer(num_units=Cfg.toy_net_width * units_mult)
            # nnet.addReLU()
            if Cfg.dropout:
                nnet.addDropoutLayer()

        if Cfg.ocsvm_loss:
            if Cfg.normalize:
                nnet.addNormLayer()
            nnet.addOCSVMLayer()
        elif Cfg.softmax_loss:
            nnet.addDenseLayer(num_units=1)
            nnet.addSigmoidLayer()
        else:
            nnet.addSVMLayer()

    def check_specific(self):

        # store primal variables on RAM
        assert Cfg.store_on_gpu
