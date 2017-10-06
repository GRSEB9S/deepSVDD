import time
import os
import numpy as np

import cPickle as pickle

from config import Configuration as Cfg


class Log(dict):

    def __init__(self, dataset_name):

        dict.__init__(self)

        self['dataset_name'] = dataset_name

        self['date_and_time'] = time.strftime('%d-%m-%Y--%H-%M-%S')

        self['time_stamp'] = []
        self['layer_tag'] = []

        self['train_objective'] = []
        self['train_accuracy'] = []
        self['train_emp_loss'] = []
        self['train_auc'] = []
        self['train_outlier_scores_summary'] = []
        self['train_normal_scores_summary'] = []
        self['train_outlier_rep_norm_summary'] = []
        self['train_normal_rep_norm_summary'] = []

        self['val_objective'] = []
        self['val_accuracy'] = []
        self['val_emp_loss'] = []
        self['val_auc'] = []
        self['val_outlier_scores_summary'] = []
        self['val_normal_scores_summary'] = []
        self['val_outlier_rep_norm_summary'] = []
        self['val_normal_rep_norm_summary'] = []

        self['test_objective'] = -1
        self['test_accuracy'] = -1

        self['l2_penalty'] = []
        self['W_svm_norm'] = []
        self['rho_svm'] = []

        self['primal_objective'] = []
        self['hinge_loss'] = []
        self['dual_objective'] = []

        for key in Cfg.__dict__:
            if key.startswith('__'):
                continue
            if key not in ('C', 'D', 'learning_rate', 'momentum', 'rho', 'nu'):
                self[key] = getattr(Cfg, key)
            else:
                self[key] = getattr(Cfg, key).get_value()

    def store_architecture(self, nnet):

        self['layers'] = dict()
        for layer in nnet.all_layers:
            self['layers'][layer.name] = dict()
            if layer.isdense or layer.issvm:
                self['layers'][layer.name]["n_in"] = \
                    np.prod(layer.input_shape[1:])
                self['layers'][layer.name]["n_out"] = layer.num_units

            if layer.isconv:
                self['layers'][layer.name]["n_filters"] = layer.num_filters
                self['layers'][layer.name]["f_size"] = layer.filter_size

            if layer.ismaxpool:
                self['layers'][layer.name]["pool_size"] = layer.pool_size

    def save_to_file(self, filename=None):

        if not filename:
            filename = '../log/all/{}-0'.format(self['date_and_time'])
            count = 1
            while os.path.exists(filename):
                filename = '../log/all/{}-{}'\
                    .format(self['date_and_time'], count)
                count += 1
            filename += '.p'

        pickle.dump(self, open(filename, 'wb'))
        print('Experiment logged in {}'.format(filename))


class AD_Log(dict):

    def __init__(self):

        dict.__init__(self)

        self['date_and_time'] = time.strftime('%d-%m-%Y--%H-%M-%S')

        self['train_auc'] = 0
        self['train_accuracy'] = 0
        self['train_time'] = 0

        self['test_auc'] = 0
        self['test_accuracy'] = 0
        self['test_time'] = 0

    def save_to_file(self, filename=None):

        with open(filename, 'wb') as f:
            pickle.dump(self, f)

        print('Anomaly detection results logged in {}'.format(filename))


def log_exp_config(xp_path, dataset):
    """
    log configuration of the experiment in a .txt-file
    """

    log_file = "{}/log.txt".format(xp_path)
    log = open(log_file, "a")

    log.write("Experiment configuration\n")
    log.write("Dataset: {}\n".format(dataset))
    log.write("Seed: {}\n".format(Cfg.seed))
    log.write("Fraction of Outliers: {}\n".format(Cfg.out_frac))
    log.write("First layer weight init by dictionary: {}\n".format(
        Cfg.weight_dict_init))
    log.write("PCA pre-processing? {}\n".format(Cfg.pca))
    log.write("Unit norm? {}\n".format(Cfg.unit_norm))
    log.write("Norm used: {}\n".format(Cfg.unit_norm_used))
    log.write("Z-Score normalization? {}\n".format(Cfg.z_normalization))
    log.write("Global contrast normalization? {}\n".format(Cfg.gcn))
    log.write("ZCA Whitening? {}\n".format(Cfg.zca_whitening))

    if dataset == 'mnist':
        str_normal = str(Cfg.mnist_normal)
        str_outlier = str(Cfg.mnist_outlier)
        if Cfg.mnist_normal == -1:
            str_normal = "all"
        if Cfg.mnist_outlier == -1:
            str_outlier = "all"
        log.write("MNIST classes: {} vs. {}\n".format(str_normal, str_outlier))
        log.write("MNIST representation dimensionality: {}\n".format(
            Cfg.mnist_rep_dim))
        log.write("MNIST Network with bias terms? {}\n".format(Cfg.mnist_bias))

    if dataset == 'cifar10':
        str_normal = str(Cfg.cifar10_normal)
        str_outlier = str(Cfg.cifar10_outlier)
        if Cfg.cifar10_normal == -1:
            str_normal = "all"
        if Cfg.cifar10_outlier == -1:
            str_outlier = "all"
        log.write("CIFAR-10 classes: {} vs. {}\n".format(str_normal,
                                                         str_outlier))
        log.write("CIFAR-10 representation dimensionality: {}\n".format(
            Cfg.cifar10_rep_dim))
        log.write("CIFAR-10 Network with bias terms? {}\n".format(
            Cfg.cifar10_bias))

    if dataset == 'bedroom':
        log.write("Total train samples used in training: {}\n".format(
            Cfg.bedroom_n_train_samples))
        log.write("Monitor interval (in train samples): {}\n".format(
            Cfg.bedroom_monitor_interval))
        log.write("Images downscaled to size: {}x{}\n".format(
            Cfg.bedroom_downscale_pxl, Cfg.bedroom_downscale_pxl))

    log.write("\n\n")
    log.close()


def log_NeuralNet(xp_path, loss, solver, learning_rate, momentum, rho, n_epochs,
                  C, B, A, nu, sparsity):
    """
    log configuration of NeuralNet-class instance
    """

    log_file = "{}/log.txt".format(xp_path)
    log = open(log_file, "a")

    log.write("NeuralNet configuration\n")
    log.write("Loss: {}\n".format(loss))
    log.write("Pre-training? {}\n".format(Cfg.pretrain))
    log.write("Solver: {}\n".format(solver))
    log.write("Learning rate: {}\n".format(learning_rate))
    log.write("Learning rate decay? {}\n".format(Cfg.lr_decay))
    log.write("Learning rate decay after epoch: {}\n".format(
        Cfg.lr_decay_after_epoch))
    log.write("Momentum: {}\n".format(momentum))
    log.write("Rho: {}\n".format(rho))
    log.write("Number of epochs: {}\n".format(n_epochs))
    log.write("Batch size: {}\n".format(Cfg.batch_size))
    log.write("Leaky ReLU: {}\n".format(Cfg.leaky_relu))
    log.write("Final layer softplus activation: {}\n\n".format(Cfg.softplus))

    log.write("Regularization\n")
    log.write("Weight decay: {}\n".format(Cfg.weight_decay))
    log.write("C-parameter: {}\n".format(C))
    log.write("Sparsity regularization: {}\n".format(Cfg.sparsity_penalty))
    log.write("Sparsity Mode: {}\n".format(Cfg.sparsity_mode))
    log.write("B-parameter (sparsity hyper-parameter): {}\n".format(B))
    log.write("Sparsity parameter (desired average activation): {}\n".format(sparsity))
    log.write("W_svm penalty: {}\n".format(Cfg.Wsvm_penalty))
    log.write("Product penalty: {}\n".format(Cfg.prod_penalty))
    log.write("Bias penalized? {}\n".format(Cfg.include_bias))
    log.write("Ball penalty: {}\n".format(Cfg.ball_penalty))
    log.write("A-parameter: {}\n".format(A))
    log.write("Output penalty: {}\n".format(Cfg.output_penalty))
    log.write("Dropout: {}\n".format(Cfg.dropout))
    log.write("Dropout architecture? {}\n\n".format(Cfg.dropout_architecture))

    if Cfg.pretrain:
        log.write("Pre-Training configuration:\n")
        log.write("Reconstruction loss: {}\n".format(Cfg.ae_loss))
        log.write("Weight decay: {}\n".format(Cfg.ae_weight_decay))
        log.write("C-parameter: {}\n".format(Cfg.ae_C.get_value()))
        log.write("Sparsity regularization: {}\n".format(
            Cfg.ae_sparsity_penalty))
        log.write("Sparsity Mode: {}\n".format(Cfg.ae_sparsity_mode))
        log.write("B-parameter (sparsity hyper-parameter): {}\n".format(
            Cfg.ae_B.get_value()))
        log.write(
            "Sparsity parameter (desired average activation): {}\n\n".format(
                Cfg.ae_sparsity.get_value()))

    if loss == 'svdd':
        log.write("SVDD\n")
        log.write("Hard margin objective? {}\n".format(Cfg.hard_margin))
        log.write("Gaussian Blob objective? {}\n".format(Cfg.gaussian_blob))
        log.write("Nu-parameter: {}\n".format(nu))
        log.write("Mean initialization of c? {}\n".format(Cfg.c_mean_init))
        log.write("Number of batches for mean initialization of c: {}\n".format(
            Cfg.c_mean_init_n_batches))
        log.write("Early compression initialization? {}\n".format(Cfg.early_compression))
        log.write("Number of epochs of early compression: {}\n".format(Cfg.ec_n_epochs))
        log.write("FastR algorithm? {}\n".format(Cfg.fastR))

    if loss == 'ocsvm':
        log.write("OC-SVM\n")
        log.write("Nu-parameter: {}\n".format(nu))
        log.write("Rho fixed? {}\n".format(Cfg.rho_fixed))
        log.write("Normalize feature map? {}\n".format(Cfg.normalize))

    if loss == 'autoencoder':
        log.write("Autoencoder\n")
        log.write("Reconstruction loss: {}\n".format(Cfg.ae_loss))
        log.write("Weight decay: {}\n".format(Cfg.ae_weight_decay))
        log.write("C-parameter: {}\n".format(Cfg.ae_C.get_value()))
        log.write("Sparsity regularization: {}\n".format(
            Cfg.ae_sparsity_penalty))
        log.write("Sparsity Mode: {}\n".format(Cfg.ae_sparsity_mode))
        log.write("B-parameter (sparsity hyper-parameter): {}\n".format(
            Cfg.ae_B.get_value()))
        log.write(
            "Sparsity parameter (desired average activation): {}\n".format(
                Cfg.ae_sparsity.get_value()))


    log.write("\n\n")
    log.close()


def log_SVM(xp_path, loss, kernel, gamma, nu):
    """
    log configuration of SVM-class instance
    """

    log_file = "{}/log.txt".format(xp_path)
    log = open(log_file, "a")

    log.write("SVM configuration\n")
    log.write("Loss: {}\n".format(loss))
    log.write("Kernel: {}\n".format(kernel))
    log.write("Gamma: {}\n".format(gamma))
    log.write("Nu-parameter: {}\n".format(nu))

    log.write("\n\n")
    log.close()


def log_KDE(xp_path, kernel, bandwidth):
    """
    log configuration of KDE-class instance
    """

    log_file = "{}/log.txt".format(xp_path)
    log = open(log_file, "a")

    log.write("KDE configuration\n")
    log.write("Kernel: {}\n".format(kernel))
    log.write("Bandwidth: {}\n".format(bandwidth))

    log.write("\n\n")
    log.close()


def log_AD_results(xp_path, learner):
    """
    log the final results to compare the performance of various learners
    """

    log_file = "{}/log.txt".format(xp_path)
    log = open(log_file, "a")

    log.write("Results\n\n")

    log.write("Train AUC: {} %\n".format(round(learner.auc_train[-1]*100, 4)))
    log.write("Train accuracy: {} %\n".format(round(learner.acc_train[-1], 4)))
    log.write("Train time: {}\n\n".format(round(learner.train_time, 4)))

    log.write("Test AUC: {} %\n".format(round(learner.auc_val[-1]*100, 4)))
    log.write("Test accuracy: {} %\n".format(round(learner.acc_val[-1], 4)))
    log.write("Test time: {}\n".format(round(learner.test_time, 4)))

    log.write("\n\n")
    log.close()
