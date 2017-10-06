import time
import os
import numpy as np
import cPickle as pickle
import theano.tensor as T
from lasagne.layers import InputLayer

import opt.sgd.train
import opt.sgd.updates

from sklearn.metrics import roc_auc_score
from datasets.main import load_dataset
from utils.monitoring import performance
from utils.misc import get_five_number_summary
from utils.pickle import dump_weights, load_weights
from utils.patches import patch_lasagne
from utils.log import Log, AD_Log
from layers import ConvLayer, ReLU, LeakyReLU, MaxPool, Upscale, DenseLayer,\
    SVMLayer, OCSVMLayer, BatchNorm, DropoutLayer, WeightNorm, Dimshuffle, \
    Reshape, Sigmoid, Softmax, Norm, Abs
from config import Configuration as Cfg


class NeuralNet:

    def __init__(self,
                 dataset,
                 use_weights=None,
                 pretrain=False,
                 profile=False):
        """ initialize instance
        """

        # whether to enable profiling in Theano functions
        self.profile = profile

        # patch lasagne creation of parameters
        # (otherwise broadcasting issue with latest versions)
        patch_lasagne()

        self.initialize_variables(dataset)

        # load dataset
        load_dataset(self, dataset.lower(), pretrain)

        if use_weights and not pretrain:
            self.load_weights(use_weights)

    def initialize_variables(self, dataset):

        self.all_layers, self.trainable_layers = (), ()

        self.n_conv_layers = 0
        self.n_dense_layers = 0
        self.n_relu_layers = 0
        self.n_leaky_relu_layers = 0
        self.n_bn_layers = 0
        self.n_norm_layers = 0
        self.n_abs_layers = 0
        self.n_maxpool_layers = 0
        self.n_upscale_layers = 0
        self.n_dropout_layers = 0
        self.n_dimshuffle_layers = 0
        self.n_reshape_layers = 0
        self.R_init = 0
        self.R_ec_seq = None

        self.learning_rate_init = Cfg.learning_rate.get_value()

        self.it = 0
        self.clock = 0

        self.log = Log(dataset_name=dataset)
        self.ad_log = AD_Log()

        self.dense_layers, self.conv_layers, = [], []

    def compile_updates(self):
        """ create network from architecture given in modules (determined by dataset)
        create Theano compiled functions
        """

        opt.sgd.updates.create_update(self)

    def compile_autoencoder(self):
        """
        create network from autoencoder architecture (determined by dataset)
        and compile Theano update functions.
        """

        print("Compiling autoencoder...")
        opt.sgd.updates.create_autoencoder(self)
        print("Autoencoder compiled.")

    def load_data(self, data_loader=None, pretrain=False):

        self.data = data_loader()

        if pretrain:
            self.data.build_autoencoder(self)

            for layer in self.all_layers:
                setattr(self, layer.name + "_layer", layer)
        elif Cfg.reconstruction_loss:
            self.data.build_autoencoder(self)

            for layer in self.all_layers:
                setattr(self, layer.name + "_layer", layer)

            self.log.store_architecture(self)
        else:
            self.data.build_architecture(self)

            for layer in self.all_layers:
                setattr(self, layer.name + "_layer", layer)

            self.log.store_architecture(self)

    def flush_data(self):

        self.data._X_train = None
        self.data._y_train = None
        self.data._X_val = None
        self.data._y_val = None
        self.data._X_test = None
        self.data._y_test = None

        print("Data flushed from network.")

    def next_layers(self, layer):

        flag = False
        for current_layer in self.all_layers:
            if flag:
                yield current_layer
            if current_layer is layer:
                flag = True

    def previous_layers(self, layer):

        flag = False
        for current_layer in reversed(self.all_layers):
            if flag:
                yield current_layer
            if current_layer is layer:
                flag = True

    def start_clock(self):

        self.clock = time.time()

    def stop_clock(self):

        self.clocked = time.time() - self.clock
        print("Total elapsed time: %g" % self.clocked)

    def pretrain(self, solver, lr, n_epochs):
        """
        pre-train weights with an autoencoder
        """

        self.ae_solver = solver.lower()
        self.ae_learning_rate = lr
        self.ae_n_epochs = n_epochs

        # set learning rate
        lr_tmp = Cfg.learning_rate.get_value()
        Cfg.learning_rate.set_value(Cfg.floatX(lr))

        self.compile_autoencoder()

        from opt.sgd.train import train_autoencoder
        train_autoencoder(self)

        # remove layer attributes, re-initialize network and reset learning rate
        for layer in self.all_layers:
            delattr(self, layer.name + "_layer")
        self.initialize_variables(self.data.dataset_name)
        Cfg.learning_rate.set_value(Cfg.floatX(lr_tmp))

        # load network architecture
        self.data.build_architecture(self)

        for layer in self.all_layers:
            setattr(self, layer.name + "_layer", layer)

        self.log.store_architecture(self)

        # load weights learned by autoencoder
        self.load_weights(Cfg.xp_path + "/ae_pretrained_weights.p")

    def train(self, solver, n_epochs=10, save_at=0, save_to=''):

        self.solver = solver.lower()
        self.ae_solver = solver.lower()
        self.n_epochs = n_epochs
        self.save_at = save_at
        self.save_to = save_to

        self.log['solver'] = self.solver
        self.log['save_at'] = self.save_at

        self.compile_updates()

        from opt.sgd.train import train_network

        self.start_clock()
        train_network(self)
        self.stop_clock()

        self.log.save_to_file()

    def evaluate(self, solver):

        # this could be simplified to only compiling the forwardpropagation...
        self.solver = solver.lower()  # needed for compiling backprop
        self.compile_updates()

        print("Evaluating network with current weights...")

        self.initialize_diagnostics(1)
        self.copy_parameters()

        # perform forward passes on training and validation set
        _, _ = performance(self, which_set='train', epoch=0, print_=True)
        _, _ = performance(self, which_set='val', epoch=0, print_=True)

        print("Evaluation on train and test set completed.")

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

    def addInputLayer(self, **kwargs):

        self.input_layer = InputLayer(name="input", **kwargs)
        self.input_layer.inp_ndim = len(kwargs["shape"])

    def addConvLayer(self, use_batch_norm=False, **kwargs):
        """
        Add convolutional layer.
        If batch norm flag is True, the convolutional layer
        will be followed by a batch-normalization layer
        """

        input_layer = self.input_layer if not self.all_layers \
            else self.all_layers[-1]

        self.n_conv_layers += 1
        name = "conv%i" % self.n_conv_layers

        new_layer = ConvLayer(input_layer, name=name, **kwargs)

        self.all_layers += (new_layer,)
        self.trainable_layers += (new_layer,)

        if use_batch_norm:
            self.n_bn_layers += 1
            name = "bn%i" % self.n_bn_layers
            self.all_layers += (BatchNorm(new_layer, name=name),)

    def addDenseLayer(self, use_batch_norm=False, normalize=False, **kwargs):
        """
        Add dense layer.
        If batch norm flag is True, the dense layer
        will be followed by a batch-normalization layer.
        If normalize flag is True, the dense layer 
        will be followed by a weight-normalization layer.
        """

        input_layer = self.input_layer if not self.all_layers \
            else self.all_layers[-1]

        self.n_dense_layers += 1
        name = "dense%i" % self.n_dense_layers

        new_layer = DenseLayer(input_layer, name=name, normalize=normalize,
                               **kwargs)

        self.all_layers += (new_layer,)
        self.trainable_layers += (new_layer,)

        if use_batch_norm:
            self.n_bn_layers += 1
            name = "bn%i" % self.n_bn_layers
            self.all_layers += (BatchNorm(new_layer, name=name),)

        if normalize:
            self.n_norm_layers += 1
            name = "norm%i" % self.n_norm_layers
            norms = T.sqrt(T.sum(new_layer.W ** 2, axis=0, dtype='floatX'))
            # Divide by the norm of the weight
            norm_layer = WeightNorm(new_layer, scales=Cfg.floatX(1.) / norms,
                                    shared_axes=0, name=name)
            # Do not optimize the scales parameter
            norm_layer.params[norm_layer.scales].remove('trainable')
            self.all_layers += (norm_layer,)

            # self.n_norm_layers += 1
            # name = "norm%i" % self.n_norm_layers
            # ceil_layer = Sigmoid(norm_layer, name=name)

    def addSVMLayer(self, **kwargs):
        """
        Add classification layer.
        """

        input_layer = self.input_layer if not self.all_layers \
            else self.all_layers[-1]
        new_layer = SVMLayer(input_layer, num_units=self.data.n_classes,
                             **kwargs)

        self.all_layers += (new_layer,)
        self.trainable_layers += (new_layer,)

        self.n_layers = len(self.all_layers)

    def addOCSVMLayer(self, **kwargs):
        """
        Add one-class SVM classification layer.
        """

        input_layer = self.input_layer if not self.all_layers \
            else self.all_layers[-1]
        new_layer = OCSVMLayer(input_layer, **kwargs)

        self.all_layers += (new_layer,)
        self.trainable_layers += (new_layer,)

        self.n_layers = len(self.all_layers)

    def addSigmoidLayer(self, **kwargs):
        """
        Add sigmoid classification layer.
        """

        input_layer = self.input_layer if not self.all_layers \
            else self.all_layers[-1]
        new_layer = Sigmoid(input_layer, **kwargs)

        self.all_layers += (new_layer,)

        self.n_layers = len(self.all_layers)

    def addSoftmaxLayer(self, **kwargs):
        """
        Add softmax multi-class classification layer.
        """

        input_layer = self.input_layer if not self.all_layers \
            else self.all_layers[-1]
        new_layer = Softmax(input_layer, **kwargs)

        self.all_layers += (new_layer,)

        self.n_layers = len(self.all_layers)

    def addNormLayer(self, **kwargs):
        """
        Add layer which normalizes its input to length 1. 
        """

        input_layer = self.input_layer if not self.all_layers \
            else self.all_layers[-1]

        self.n_norm_layers += 1
        name = "norm%i" % self.n_norm_layers

        new_layer = Norm(input_layer, name=name, **kwargs)

        self.all_layers += (new_layer,)

    def addAbsLayer(self, **kwargs):
        """
        Add layer which returns the absolute value of its input.
        """

        input_layer = self.input_layer if not self.all_layers \
            else self.all_layers[-1]

        self.n_abs_layers += 1
        name = "abs%i" % self.n_abs_layers

        new_layer = Abs(input_layer, name=name, **kwargs)

        self.all_layers += (new_layer,)

    def addReLU(self, **kwargs):
        """
        Add ReLU activation layer.
        """

        input_layer = self.input_layer if not self.all_layers \
            else self.all_layers[-1]

        self.n_relu_layers += 1
        name = "relu%i" % self.n_relu_layers

        new_layer = ReLU(input_layer, name=name, **kwargs)

        self.all_layers += (new_layer,)

    def addLeakyReLU(self, **kwargs):
        """
        Add leaky ReLU activation layer. (with leakiness=0.01)
        """

        input_layer = self.input_layer if not self.all_layers \
            else self.all_layers[-1]

        self.n_leaky_relu_layers += 1
        name = "leaky_relu%i" % self.n_leaky_relu_layers

        new_layer = LeakyReLU(input_layer, name=name, **kwargs)

        self.all_layers += (new_layer,)

    def addMaxPool(self, **kwargs):
        """
        Add MaxPooling activation layer.
        """

        input_layer = self.input_layer if not self.all_layers\
            else self.all_layers[-1]

        self.n_maxpool_layers += 1
        name = "maxpool%i" % self.n_maxpool_layers

        new_layer = MaxPool(input_layer, name=name, **kwargs)

        self.all_layers += (new_layer,)

    def addUpscale(self, **kwargs):
        """
        Add Upscaling activation layer.
        """

        input_layer = self.input_layer if not self.all_layers\
            else self.all_layers[-1]

        self.n_upscale_layers += 1
        name = "upscale%i" % self.n_upscale_layers

        new_layer = Upscale(input_layer, name=name, **kwargs)

        self.all_layers += (new_layer,)

    def addDropoutLayer(self, **kwargs):
        """
        Add Dropout layer.
        """

        input_layer = self.input_layer if not self.all_layers \
            else self.all_layers[-1]

        self.n_dropout_layers += 1
        name = "dropout%i" % self.n_dropout_layers

        new_layer = DropoutLayer(input_layer, name=name, **kwargs)

        self.all_layers += (new_layer,)

    def addDimshuffleLayer(self, **kwargs):
        """
        Add Dimshuffle layer to reorder dimensions
        """

        input_layer = self.input_layer if not self.all_layers \
            else self.all_layers[-1]

        self.n_dimshuffle_layers += 1
        name = "dimshuffle%i" % self.n_dimshuffle_layers

        new_layer = Dimshuffle(input_layer, name=name, **kwargs)

        self.all_layers += (new_layer,)

    def addReshapeLayer(self, **kwargs):
        """
        Add reshape layer to reshape dimensions
        """

        input_layer = self.input_layer if not self.all_layers \
            else self.all_layers[-1]

        self.n_reshape_layers += 1
        name = "reshape%i" % self.n_reshape_layers

        new_layer = Reshape(input_layer, name=name, **kwargs)

        self.all_layers += (new_layer,)

    def dump_weights(self, filename=None, pretrain=False):

        dump_weights(self, filename, pretrain=pretrain)

    def load_weights(self, filename=None):

        assert filename and os.path.exists(filename)

        load_weights(self, filename)

    def initialize_diagnostics(self, n_epochs):
        """
        initialize attributes for diagnostics on the network and method
        """

        # data-dependent values
        self.acc_train = np.zeros(n_epochs, dtype=Cfg.floatX)
        self.acc_val = np.zeros(n_epochs, dtype=Cfg.floatX)

        self.objective_train = np.zeros(n_epochs, dtype=Cfg.floatX)
        self.objective_val = np.zeros(n_epochs, dtype=Cfg.floatX)
        self.sparsity_penalty_train = np.zeros(n_epochs, dtype=Cfg.floatX)
        self.sparsity_penalty_val = np.zeros(n_epochs, dtype=Cfg.floatX)
        self.ball_penalty_train = np.zeros(n_epochs, dtype=Cfg.floatX)
        self.ball_penalty_val = np.zeros(n_epochs, dtype=Cfg.floatX)
        self.output_penalty_train = np.zeros(n_epochs, dtype=Cfg.floatX)
        self.output_penalty_val = np.zeros(n_epochs, dtype=Cfg.floatX)
        self.emp_loss_train = np.zeros(n_epochs, dtype=Cfg.floatX)
        self.emp_loss_val = np.zeros(n_epochs, dtype=Cfg.floatX)

        # hacky extraction of monitoring values for bedroom dataset
        if self.data.dataset_name == "bedroom":
            self.scores_train = np.zeros((len(self.data._y_val), n_epochs),
                                         dtype=Cfg.floatX)
            self.scores_val = np.zeros((len(self.data._y_test), n_epochs),
                                       dtype=Cfg.floatX)
            self.rep_norm_train = np.zeros((len(self.data._y_val), n_epochs),
                                           dtype=Cfg.floatX)
            self.rep_norm_val = np.zeros((len(self.data._y_test), n_epochs),
                                         dtype=Cfg.floatX)
        else:
            self.scores_train = np.zeros((len(self.data._y_train), n_epochs),
                                         dtype=Cfg.floatX)
            self.scores_val = np.zeros((len(self.data._y_val), n_epochs),
                                       dtype=Cfg.floatX)
            self.rep_norm_train = np.zeros((len(self.data._y_train), n_epochs),
                                           dtype=Cfg.floatX)
            self.rep_norm_val = np.zeros((len(self.data._y_val), n_epochs),
                                         dtype=Cfg.floatX)

        self.auc_train = np.zeros(n_epochs, dtype=Cfg.floatX)
        self.auc_val = np.zeros(n_epochs, dtype=Cfg.floatX)

        if Cfg.ocsvm_loss:
            self.rep_train = np.zeros(self.data._X_train.shape, dtype=Cfg.floatX)
            self.rep_val = np.zeros(self.data._X_val.shape, dtype=Cfg.floatX)

        # network parameters
        self.l2_penalty = np.zeros(n_epochs, dtype=Cfg.floatX)

        self.W_norms = []
        self.b_norms = []
        self.dW_norms = []
        self.db_norms = []
        self.W_copy = []
        self.b_copy = []
        for layer in self.trainable_layers:
            if layer.isdense:
                self.W_norms.append(np.zeros((layer.num_units, n_epochs),
                                             dtype=Cfg.floatX))
                if layer.b is None:
                    self.b_norms.append(None)
                else:
                    self.b_norms.append(
                        np.zeros((layer.num_units, n_epochs),
                                 dtype=Cfg.floatX))

            if layer.isdense | layer.isconv:
                self.dW_norms.append(np.zeros(n_epochs, dtype=Cfg.floatX))
                if layer.b is None:
                    self.db_norms.append(None)
                else:
                    self.db_norms.append(np.zeros(n_epochs,
                                                  dtype=Cfg.floatX))
                self.W_copy.append(None)
                self.b_copy.append(None)

        # Best results (highest AUC on test set)
        self.auc_best = 0
        self.auc_best_epoch = 0  # determined by highest AUC on test set
        self.best_weight_dict = None

        # OC-SVM layer parameters
        self.Wsvm_norm = np.zeros(n_epochs, dtype=Cfg.floatX)
        self.rhosvm = np.zeros(n_epochs, dtype=Cfg.floatX)
        self.dWsvm_norm = np.zeros(n_epochs, dtype=Cfg.floatX)
        self.drhosvm_norm = np.zeros(n_epochs, dtype=Cfg.floatX)
        self.Wsvm_copy = None
        self.rhosvm_copy = None

        # SVDD parameters
        self.R = np.zeros(n_epochs, dtype=Cfg.floatX)

    def save_objective_and_accuracy(self, epoch, which_set,
                                    objective, accuracy):
        """
        save objective and accuracy of epoch
        """

        if which_set == 'train':
            self.objective_train[epoch] = objective
            self.acc_train[epoch] = accuracy

        if which_set == 'val':
            self.objective_val[epoch] = objective
            self.acc_val[epoch] = accuracy

    def save_initial_parameters(self):
        """
        save a copy of the initial network parameters for diagnostics.
        """

        self.W_init = []
        self.b_init = []

        for layer in self.trainable_layers:
            if layer.isdense | layer.isconv:
                self.W_init.append(None)
                self.b_init.append(None)

        i = 0
        for layer in self.trainable_layers:
            if layer.isdense | layer.isconv:
                self.W_init[i] = layer.W.get_value()
                if layer.b is not None:
                    self.b_init[i] = layer.b.get_value()
                i += 1

        if Cfg.ocsvm_loss:
            self.Wsvm_init = self.ocsvm_layer.W.get_value()
            if not Cfg.rho_fixed:
                self.rhosvm_init = self.ocsvm_layer.b.get_value()

    def copy_parameters(self):
        """
        save a copy of the current network parameters in order to monitor the
         difference between epochs.
        """
        i = 0
        for layer in self.trainable_layers:
            if layer.isdense | layer.isconv:
                self.W_copy[i] = layer.W.get_value()
                if layer.b is not None:
                    self.b_copy[i] = layer.b.get_value()
                i += 1

        if Cfg.ocsvm_loss:
            self.Wsvm_copy = self.ocsvm_layer.W.get_value()
            if not Cfg.rho_fixed:
                self.rhosvm_copy = self.ocsvm_layer.b.get_value()

    def copy_initial_parameters_to_cache(self):
        """
        Save a copy of the initial parameters in cache
        """
        self.W_copy = list(self.W_init)
        self.b_copy = list(self.b_init)

        if Cfg.ocsvm_loss:
            self.Wsvm_copy = list(self.Wsvm_init)
            if not Cfg.rho_fixed:
                self.rhosvm_copy = list(self.rhosvm_init)

    def save_network_diagnostics(self, epoch, l2, R):
        """
        save diagnostics of the network
        """

        self.l2_penalty[epoch] = l2
        self.log['l2_penalty'].append(float(l2))

        i = 0
        j = 0
        for layer in self.trainable_layers:
            if layer.isdense:
                self.W_norms[i][:, epoch] = np.sum(layer.W.get_value() ** 2,
                                                   axis=0)
                if layer.b is not None:
                    self.b_norms[i][:, epoch] = layer.b.get_value() ** 2
                i += 1

            if layer.isdense | layer.isconv:
                dW = np.sqrt(np.sum(
                    (layer.W.get_value() - self.W_copy[j]) ** 2))
                self.dW_norms[j][epoch] = dW
                if layer.b is not None:
                    db = np.sqrt(np.sum(
                        (layer.b.get_value() - self.b_copy[j]) ** 2))
                    self.db_norms[j][epoch] = db
                j += 1

        # diagnostics only relevant for the oc-svm loss
        if Cfg.ocsvm_loss:
            W_svm_norm = float(np.sum(self.ocsvm_layer.W.get_value() ** 2))
            if Cfg.rho_fixed:
                W_svm_norm *= 0.5 * (1.0 / Cfg.C.get_value())
            else:
                W_svm_norm *= 0.5
            self.Wsvm_norm[epoch] = W_svm_norm
            self.log['W_svm_norm'].append(W_svm_norm)

            if not Cfg.rho_fixed:
                rho_svm = float(np.sum(self.ocsvm_layer.b.get_value()))
                self.rhosvm[epoch] = rho_svm
                self.log['rho_svm'].append(rho_svm)

            dWsvm = np.sqrt(np.sum(
                (self.ocsvm_layer.W.get_value() - self.Wsvm_copy) ** 2))
            self.dWsvm_norm[epoch] = dWsvm
            if not Cfg.rho_fixed:
                drhosvm = np.sqrt(np.sum(
                    (self.ocsvm_layer.b.get_value() - self.rhosvm_copy) ** 2))
                self.drhosvm_norm[epoch] = drhosvm

        # diagnostics only relevant for the svdd loss
        if Cfg.svdd_loss:
            self.R[epoch] = R

    def track_best_results(self, epoch):
        """
        Save network parameters where AUC on the validation set was highest.
        """

        if self.auc_val[epoch] > self.auc_best:
            self.auc_best = self.auc_val[epoch]
            self.auc_best_epoch = epoch

            self.best_weight_dict = dict()

            for layer in self.trainable_layers:
                self.best_weight_dict[layer.name + "_w"] = layer.W.get_value()
                if layer.b is not None:
                    self.best_weight_dict[layer.name + "_b"] = layer.b.get_value()

            if Cfg.svdd_loss:
                self.best_weight_dict["R"] = self.Rvar.get_value()

    def dump_best_weights(self, filename):
        """
        pickle the network parameters, where AUC on the test set was highest.
        """

        with open(filename, 'wb') as f:
            pickle.dump(self.best_weight_dict, f)

        print("Parameters of best epoch saved in %s" % filename)

    def save_train_diagnostics(self, epoch, scores, rep_norm, rep, emp_loss,
                               sparsity_penalty, ball_penalty, output_penalty):
        """
        save training set diagnostics of epoch
        """
        train_scores = scores.flatten()

        if self.data.n_classes == 2:

            if Cfg.ocsvm_loss | Cfg.svm_loss:

                if Cfg.ocsvm_loss:
                    train_scores = -train_scores  # greater scores indicate outliers

                    self.rep_train = rep

                    self.ball_penalty_train[epoch] = ball_penalty
                    self.output_penalty_train[epoch] = output_penalty

            self.scores_train[:, epoch] = train_scores

            # hacky extraction of monitoring values for bedroom dataset
            if self.data.dataset_name == "bedroom":
                train_scores_normal = train_scores[self.data._y_val == 0]
                train_scores_outlier = train_scores[self.data._y_val == 1]
                rep_norm_train_normal = rep_norm[self.data._y_val == 0]
                rep_norm_train_outlier = rep_norm[self.data._y_val == 1]

                if sum(self.data._y_val) > 0:
                    AUC = roc_auc_score(self.data._y_val, train_scores)
                    self.auc_train[epoch] = AUC
                    self.log['train_auc'].append(float(AUC))
                    print("{:32} {:.2f}%".format('AUC:', 100. * AUC))
            else:
                train_scores_normal = train_scores[self.data._y_train == 0]
                train_scores_outlier = train_scores[self.data._y_train == 1]
                rep_norm_train_normal = rep_norm[self.data._y_train == 0]
                rep_norm_train_outlier = rep_norm[self.data._y_train == 1]

                if sum(self.data._y_train) > 0:
                    AUC = roc_auc_score(self.data._y_train, train_scores)
                    self.auc_train[epoch] = AUC
                    self.log['train_auc'].append(float(AUC))
                    print("{:32} {:.2f}%".format('AUC:', 100. * AUC))

            normal_summary = get_five_number_summary(train_scores_normal)
            outlier_summary = get_five_number_summary(train_scores_outlier)
            self.log['train_normal_scores_summary'].append(normal_summary)
            self.log['train_outlier_scores_summary'].append(outlier_summary)

            self.rep_norm_train[:, epoch] = rep_norm
            normal_summary = get_five_number_summary(rep_norm_train_normal)
            outlier_summary = get_five_number_summary(rep_norm_train_outlier)
            self.log['train_normal_rep_norm_summary'].append(normal_summary)
            self.log['train_outlier_rep_norm_summary'].append(outlier_summary)

        self.sparsity_penalty_train[epoch] = sparsity_penalty
        self.emp_loss_train[epoch] = float(emp_loss)
        self.log['train_emp_loss'].append(float(emp_loss))


    def save_val_diagnostics(self, epoch, scores, rep_norm, rep, emp_loss,
                             sparsity_penalty, ball_penalty, output_penalty):
        """
        save validation set diagnostics of epoch
        """
        val_scores = scores.flatten()

        if self.data.n_classes == 2:

            if Cfg.ocsvm_loss | Cfg.svm_loss:
                self.rep_norm_val[:, epoch] = rep_norm

                rep_norm_val_normal = rep_norm[self.data._y_val == 0]
                rep_norm_val_outlier = rep_norm[self.data._y_val == 1]
                normal_summary = get_five_number_summary(rep_norm_val_normal)
                outlier_summary = get_five_number_summary(rep_norm_val_outlier)
                self.log['val_normal_rep_norm_summary'].append(normal_summary)
                self.log['val_outlier_rep_norm_summary'].append(outlier_summary)

                if Cfg.ocsvm_loss:
                    val_scores = -val_scores  # greater scores indicate outliers

                    self.rep_val = rep

                    self.ball_penalty_val[epoch] = ball_penalty
                    self.output_penalty_val[epoch] = output_penalty

            self.scores_val[:, epoch] = val_scores

            # hacky extraction of monitoring values for bedroom dataset
            if self.data.dataset_name == "bedroom":
                val_scores_normal = val_scores[self.data._y_test == 0]
                val_scores_outlier = val_scores[self.data._y_test == 1]
                rep_norm_val_normal = rep_norm[self.data._y_test == 0]
                rep_norm_val_outlier = rep_norm[self.data._y_test == 1]

                AUC = roc_auc_score(self.data._y_test, val_scores)
                self.auc_val[epoch] = AUC
                self.log['val_auc'].append(float(AUC))
                print("{:32} {:.2f}%".format('AUC:', 100. * AUC))
            else:
                val_scores_normal = val_scores[self.data._y_val == 0]
                val_scores_outlier = val_scores[self.data._y_val == 1]
                rep_norm_val_normal = rep_norm[self.data._y_val == 0]
                rep_norm_val_outlier = rep_norm[self.data._y_val == 1]

                AUC = roc_auc_score(self.data._y_val, val_scores)
                self.auc_val[epoch] = AUC
                self.log['val_auc'].append(float(AUC))
                print("{:32} {:.2f}%".format('AUC:', 100. * AUC))

            normal_summary = get_five_number_summary(val_scores_normal)
            outlier_summary = get_five_number_summary(val_scores_outlier)
            self.log['val_normal_scores_summary'].append(normal_summary)
            self.log['val_outlier_scores_summary'].append(outlier_summary)

            self.rep_norm_val[:, epoch] = rep_norm
            normal_summary = get_five_number_summary(rep_norm_val_normal)
            outlier_summary = get_five_number_summary(rep_norm_val_outlier)
            self.log['val_normal_rep_norm_summary'].append(normal_summary)
            self.log['val_outlier_rep_norm_summary'].append(outlier_summary)

        self.sparsity_penalty_val[epoch] = sparsity_penalty
        self.emp_loss_val[epoch] = float(emp_loss)
        self.log['val_emp_loss'].append(float(emp_loss))

    def initialize_ae_diagnostics(self):
        """
        initialize attributes for diagnostics on autoencoder network
        """

        n_epochs = self.ae_n_epochs

        self.train_time = 0
        self.test_time = 0
        self.best_weight_dict = None

        # data-dependent values
        self.objective_train = np.zeros(n_epochs, dtype=Cfg.floatX)
        self.objective_val = np.zeros(n_epochs, dtype=Cfg.floatX)
        self.emp_loss_train = np.zeros(n_epochs, dtype=Cfg.floatX)
        self.emp_loss_val = np.zeros(n_epochs, dtype=Cfg.floatX)
        self.sparsity_penalty_train = np.zeros(n_epochs, dtype=Cfg.floatX)
        self.sparsity_penalty_val = np.zeros(n_epochs, dtype=Cfg.floatX)
        self.scores_train = np.zeros((len(self.data._y_train), n_epochs),
                                     dtype=Cfg.floatX)
        self.scores_val = np.zeros((len(self.data._y_val), n_epochs),
                                   dtype=Cfg.floatX)
        self.auc_train = np.zeros(n_epochs, dtype=Cfg.floatX)
        self.auc_val = np.zeros(n_epochs, dtype=Cfg.floatX)
        self.acc_train = np.zeros(n_epochs, dtype=Cfg.floatX)
        self.acc_val = np.zeros(n_epochs, dtype=Cfg.floatX)

        # network parameters
        self.l2_penalty = np.zeros(n_epochs, dtype=Cfg.floatX)

    def save_ae_diagnostics(self, epoch, train_err, val_err, train_sparse,
                                  val_sparse, train_scores, val_scores, l2):
        """
        save autoencoder diagnostics
        """

        self.objective_train[epoch] = train_err + l2 + train_sparse
        self.objective_val[epoch] = val_err + l2 + val_sparse
        self.emp_loss_train[epoch] = train_err
        self.emp_loss_val[epoch] = val_err
        self.sparsity_penalty_train[epoch] = train_sparse
        self.sparsity_penalty_val[epoch] = val_sparse
        self.scores_train[:, epoch] = train_scores
        self.scores_val[:, epoch] = val_scores

        if sum(self.data._y_train) > 0:
            AUC = roc_auc_score(self.data._y_train, train_scores)
            self.auc_train[epoch] = AUC

        if sum(self.data._y_val) > 0:
            AUC = roc_auc_score(self.data._y_val, val_scores)
            self.auc_val[epoch] = AUC

        self.l2_penalty[epoch] = l2
