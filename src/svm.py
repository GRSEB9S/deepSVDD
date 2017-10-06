import time
import os
import numpy as np
from sklearn import svm
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import pairwise_distances

from datasets.main import load_dataset
from kernels import degree_kernel, weighted_degree_kernel
from config import Configuration as Cfg
from utils.log import AD_Log
from utils.pickle import dump_svm, load_svm


class SVM(object):

    def __init__(self, loss, dataset, kernel, **kwargs):

        # initialize
        self.svm = None
        self.loss = loss
        self.kernel = kernel
        self.K_train = None
        self.K_val = None
        self.K_test = None
        self.initialize_svm(loss, **kwargs)

        # load dataset
        load_dataset(self, dataset)

        # train and test time
        self.clock = 0
        self.clocked = 0
        self.train_time = 0
        self.test_time = 0

        # Scores and AUC
        self.scores_train = np.zeros((len(self.data._y_train), 1))
        self.scores_val = np.zeros((len(self.data._y_val), 1))
        self.auc_train = np.zeros(1)
        self.auc_val = np.zeros(1)
        self.acc_train = np.zeros(1)
        self.acc_val = np.zeros(1)
        self.rho = None

        # AD results log
        self.ad_log = AD_Log()

        # diagnostics
        self.best_weight_dict = None  # attribute to reuse nnet plot-functions

    def initialize_svm(self, loss, **kwargs):

        assert loss in ('SVC', 'OneClassSVM')

        if self.kernel in ('linear', 'poly', 'rbf', 'sigmoid'):
            kernel = self.kernel
        else:
            kernel = 'precomputed'

        if loss == 'SVC':
            self.svm = svm.SVC(kernel=kernel, C=Cfg.svm_C, **kwargs)
        if loss == 'OneClassSVM':
            self.svm = svm.OneClassSVM(kernel=kernel, nu=Cfg.svm_nu, **kwargs)

    def load_data(self, data_loader=None, pretrain=False):

        self.data = data_loader()

    def flush_data(self):

        self.data._X_train = None
        self.data._y_train = None
        self.data._X_val = None
        self.data._y_val = None
        self.data._X_test = None
        self.data._y_test = None

        print("Data flushed from model.")

    def start_clock(self):

        self.clock = time.time()

    def stop_clock(self):

        self.clocked = time.time() - self.clock
        print("Total elapsed time: %g" % self.clocked)

    def train(self, **kwargs):

        if self.data._X_train.ndim > 2:
            X_train_shape = self.data._X_train.shape
            X_train = self.data._X_train.reshape(X_train_shape[0],
                                                 np.prod(X_train_shape[1:]))
        else:
            X_train = self.data._X_train

        print("Starting training...")
        self.start_clock()

        if self.loss == 'SVC':

            if self.kernel in ('DegreeKernel', 'WeightedDegreeKernel'):
                self.get_kernel_matrix(kernel=self.kernel, which_set='train',
                                       **kwargs)
                self.svm.fit(self.K_train, self.data._y_train)
            else:
                self.svm.fit(X_train, self.data._y_train)

        if self.loss == 'OneClassSVM':

            if self.kernel in ('DegreeKernel', 'WeightedDegreeKernel'):
                self.get_kernel_matrix(kernel=self.kernel, which_set='train',
                                       **kwargs)
                self.svm.fit(self.K_train)
            else:
                # if rbf-kernel, re-initialize svm with gamma minimizing the
                # numerical error
                if self.kernel == 'rbf':
                    gamma = 1 / (np.max(pairwise_distances(X_train)) ** 2)
                    self.svm = svm.OneClassSVM(kernel='rbf', nu=Cfg.svm_nu,
                                               gamma=gamma)

                self.svm.fit(X_train)

        self.stop_clock()
        self.train_time = self.clocked

    def predict(self, which_set='train', **kwargs):

        assert which_set in ('train', 'val')

        if which_set == 'train':
            X = self.data._X_train
            y = self.data._y_train
        if which_set == 'val':
            X = self.data._X_val
            y = self.data._y_val

        # reshape to 2D if input is tensor
        if X.ndim > 2:
            X_shape = X.shape
            X = X.reshape(X_shape[0], np.prod(X_shape[1:]))

        print("Starting prediction...")
        self.start_clock()

        if self.loss == 'SVC':

            if self.kernel in ('DegreeKernel', 'WeightedDegreeKernel'):
                self.get_kernel_matrix(kernel=self.kernel, which_set=which_set,
                                       **kwargs)
                if which_set == 'train':
                    scores = self.svm.decision_function(self.K_train)
                if which_set == 'val':
                    scores = self.svm.decision_function(self.K_val)
            else:
                scores = self.svm.decision_function(X)

            auc = roc_auc_score(y, scores[:, 0])

            if which_set == 'train':
                self.scores_train = scores
                self.auc_train[0] = auc
            if which_set == 'val':
                self.scores_val = scores
                self.auc_val[0] = auc

        if self.loss == 'OneClassSVM':

            if self.kernel in ('DegreeKernel', 'WeightedDegreeKernel'):
                self.get_kernel_matrix(kernel=self.kernel, which_set=which_set,
                                       **kwargs)
                if which_set == 'train':
                    scores = (-1.0) * self.svm.decision_function(self.K_train)
                    y_pred = (self.svm.predict(self.K_train) == -1) * 1
                if which_set == 'val':
                    scores = (-1.0) * self.svm.decision_function(self.K_val)
                    y_pred = (self.svm.predict(self.K_val) == -1) * 1
            else:
                scores = (-1.0) * self.svm.decision_function(X)
                y_pred = (self.svm.predict(X) == -1) * 1

            if which_set == 'train':
                self.scores_train[:, 0] = scores.flatten()
                self.acc_train[0] = 100.0 * sum(y == y_pred) / len(y)
            if which_set == 'val':
                self.scores_val[:, 0] = scores.flatten()
                self.acc_val[0] = 100.0 * sum(y == y_pred) / len(y)

            if sum(y) > 0:
                auc = roc_auc_score(y, scores.flatten())
                if which_set == 'train':
                    self.auc_train[0] = auc
                if which_set == 'val':
                    self.auc_val[0] = auc

            self.rho = -self.svm.intercept_[0]

        self.stop_clock()
        if which_set == 'val':
            self.test_time = self.clocked

    def dump_model(self, filename=None):

        dump_svm(self, filename)

    def load_model(self, filename=None):

        assert filename and os.path.exists(filename)

        load_svm(self, filename)

    def log_results(self, filename=None):
        """
        log the results relevant for anomaly detection
        """

        self.ad_log['train_auc'] = self.auc_train[-1]
        self.ad_log['train_accuracy'] = self.acc_train[-1]
        self.ad_log['train_time'] = self.train_time

        self.ad_log['test_auc'] = self.auc_val[-1]
        self.ad_log['test_accuracy'] = self.acc_val[-1]
        self.ad_log['test_time'] = self.test_time

        self.ad_log.save_to_file(filename=filename)


    def get_kernel_matrix(self, kernel, which_set='train', **kwargs):

        assert kernel in ('DegreeKernel', 'WeightedDegreeKernel')

        if kernel == 'DegreeKernel':
            kernel_function = degree_kernel
        if kernel == 'WeightedDegreeKernel':
            kernel_function = weighted_degree_kernel

        if which_set == 'train':
            self.K_train = kernel_function(self.data._X_train,
                                           self.data._X_train, **kwargs)
        if which_set == 'val':
            self.K_val = kernel_function(self.data._X_val, self.data._X_train,
                                         **kwargs)
        if which_set == 'test':
            self.K_test = kernel_function(self.data._X_test, self.data._X_train,
                                          **kwargs)
