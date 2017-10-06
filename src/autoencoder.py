import time
import numpy as np
import theano
import theano.tensor as T
import lasagne.layers
import lasagne.nonlinearities

from sklearn import svm
from sklearn.metrics import roc_auc_score
from lasagne.regularization import regularize_network_params, l2
from datasets.main import load_dataset
from datasets.iterator import iterate_batches
from config import Configuration as Cfg


class AutoencoderModel(object):

    def __init__(self, dataset):

        # initialize variables
        self.encoder = None
        self.decoder = None
        self.encoding_size = 16
        self.learned_code_train = None
        self.learned_code_val = None

        self.ocsvm = None
        self.nu = theano.shared(Cfg.floatX(Cfg.svm_nu), name='nu')
        self.ocsvm_W = theano.shared(
            np.zeros((16, 1), dtype=Cfg.floatX), name='ocsvm_W')
        self.ocsvm_rho = theano.shared(Cfg.floatX(0), name='ocsvm_rho')

        # initialize scores and auc
        self.scores_train = None
        self.scores_val = None
        self.auc_train = None
        self.auc_val = None
        self.rho = None

        # load dataset
        load_dataset(self, dataset)

    def load_data(self, data_loader=None):

        self.data = data_loader()
        self.data.build_autoencoder(self)

    def flush_data(self):

        self.data._X_train = None
        self.data._y_train = None
        self.data._X_val = None
        self.data._y_val = None
        self.data._X_test = None
        self.data._y_test = None

        print("Data flushed from model.")

    def compile_network_updates(self):

        print('Compiling Theano functions...')
        self.compile_autoencoder()
        self.compile_ocsvm()
        self.compile_prediction()
        print('Theano functions compiled.')

    def compile_autoencoder(self):

        input_var = T.tensor4(name='inputs')
        target_var = T.tensor4(name='targets')

        # training
        trainable_params = lasagne.layers.get_all_params(self.decoder,
                                                         trainable=True)
        prediction = lasagne.layers.get_output(self.decoder, inputs=input_var,
                                               deterministic=False)

        # autoencoder loss: reconstruction error (i.e. squared error loss)
        loss_autoenc = lasagne.objectives.squared_error(prediction,
                                                        target_var)
        loss_autoenc = T.mean(T.sum(loss_autoenc, axis=(1, 2, 3)))
        l2_penalty = (Cfg.floatX(0.5)/Cfg.C)*regularize_network_params(
            self.decoder, l2)
        loss_autoenc = loss_autoenc + l2_penalty
        updates_autoenc = lasagne.updates.adagrad(
            loss_autoenc, trainable_params, learning_rate=Cfg.learning_rate)

        self.train_network_autoenc = theano.function([input_var, target_var],
                                                     loss_autoenc,
                                                     updates=updates_autoenc)

        # validation
        val_prediction = lasagne.layers.get_output(self.decoder,
                                                   inputs=input_var,
                                                   deterministic=True)

        val_loss_autoenc = lasagne.objectives.squared_error(val_prediction,
                                                            target_var)
        val_loss_autoenc = T.mean(T.sum(val_loss_autoenc, axis=(1, 2, 3)))
        val_loss_autoenc =val_loss_autoenc + l2_penalty

        self.val_network_autoenc = theano.function([input_var, target_var],
                                                   val_loss_autoenc)

    def compile_ocsvm(self):

        input_var = T.tensor4(name='inputs')

        # training
        trainable_params = lasagne.layers.get_all_params(self.encoder,
                                                         trainable=True)
        prediction = lasagne.layers.get_output(self.encoder, inputs=input_var,
                                               deterministic=False)

        l2_penalty = (Cfg.floatX(0.5)/Cfg.C)*regularize_network_params(
            self.encoder, l2)

        # one-class SVM loss (given one-class SVM parameters)
        scores = T.dot(prediction, self.ocsvm_W)
        scores = scores - T.fill(scores, self.ocsvm_rho)
        loss_ocsvm = T.max(T.stack([-scores, T.zeros_like(scores)], axis=1),
                           axis=1)
        loss_ocsvm = T.sum(loss_ocsvm) / (scores.shape[0] * self.nu)
        loss_ocsvm = l2_penalty + loss_ocsvm - self.ocsvm_rho

        updates_ocsvm = lasagne.updates.adagrad(
            loss_ocsvm, trainable_params, learning_rate=Cfg.learning_rate)

        self.train_network_ocsvm = theano.function([input_var], loss_ocsvm,
                                                   updates=updates_ocsvm)

        # validation
        val_prediction = lasagne.layers.get_output(self.encoder,
                                                   inputs=input_var,
                                                   deterministic=True)

        val_scores = T.dot(val_prediction, self.ocsvm_W)
        val_scores = val_scores - T.fill(val_scores, self.ocsvm_rho)
        val_loss_ocsvm = T.max(T.stack([-val_scores, T.zeros_like(val_scores)],
                                       axis=1),
                               axis=1)
        val_loss_ocsvm = T.sum(val_loss_ocsvm) / (val_scores.shape[0] * self.nu)
        val_loss_ocsvm = l2_penalty + val_loss_ocsvm - self.ocsvm_rho

        self.val_network_ocsvm = theano.function([input_var], val_loss_ocsvm)

    def compile_prediction(self):

        input_var = T.tensor4(name='inputs')
        prediction = lasagne.layers.get_output(self.encoder, inputs=input_var,
                                               deterministic=True)

        # get feature vector
        self.get_network_prediction = theano.function([input_var], prediction)

    def train(self, n_iter, n_epochs_auto=200, n_epochs_adj=1, kernel='linear'):

        self.compile_network_updates()
        self.initialize_ocsvm(kernel=kernel)

        # Train autoencoder
        self.train_autoencoder(n_epochs=n_epochs_auto)
        self.get_learned_features()
        # self.normalize_feature_vecs()

        # One-class SVM weight initialization
        print("Weight initialization:")
        self.train_ocsvm()

        print("")
        print("Begin iterative learning:")

        self.scores_train = np.zeros((len(self.data._y_train), n_iter),
                                     dtype=Cfg.floatX)
        self.scores_val = np.zeros((len(self.data._y_val), n_iter),
                                   dtype=Cfg.floatX)
        self.auc_train = np.zeros(n_iter, dtype=Cfg.floatX)
        self.auc_val = np.zeros(n_iter, dtype=Cfg.floatX)
        self.rho = np.zeros(n_iter, dtype=Cfg.floatX)

        for i in range(n_iter):
            print("Iteration {} of {}:".format(i + 1, n_iter))

            # Train network on ocsvm loss
            self.train_encoder_ocsvm(n_epochs=n_epochs_adj)

            # Get learned features (and normalize)
            self.get_learned_features()
            # base_model.normalize_feature_vecs()  # normalize learned features

            # Train one-class SVM on network output
            self.train_ocsvm()

            # Save results
            self.scores_train[:, i] = self.ocsvm_scores_train.flatten()
            self.scores_val[:, i] = self.ocsvm_scores_val.flatten()
            self.auc_train[i] = self.ocsvm_auc_train
            self.auc_val[i] = self.ocsvm_auc_val
            self.rho[i] = self.ocsvm_rho.get_value()

    def train_autoencoder(self, n_epochs=200):

        for epoch in range(n_epochs):

            # In each epoch, we do a full pass over the training data:
            train_err = 0
            train_batches = 0
            start_time = time.time()

            for batch in iterate_batches(self.data._X_train, self.data._X_train,
                                         Cfg.batch_size, shuffle=True):

                inputs, targets, _ = batch
                err = self.train_network_autoenc(inputs, targets)
                train_err += err
                train_batches += 1

            # And a full pass over the validation data:
            val_err = 0
            val_batches = 0

            for batch in iterate_batches(self.data._X_val, self.data._X_val,
                                         Cfg.batch_size, shuffle=False):

                inputs, targets, _ = batch
                err = self.val_network_autoenc(inputs, targets)
                val_err += err
                val_batches += 1

            # print results for epoch
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, n_epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err/train_batches))
            print("  validation loss:\t\t{:.6f}".format(val_err/val_batches))

    def train_encoder_ocsvm(self, n_epochs=1):

        for epoch in range(n_epochs):

            # In each epoch, we do a full pass over the training data:
            train_err = 0
            train_batches = 0
            # start_time = time.time()

            for batch in iterate_batches(self.data._X_train, self.data._y_train,
                                         Cfg.batch_size, shuffle=True):

                inputs, _, _ = batch
                err = self.train_network_ocsvm(inputs)
                train_err += err
                train_batches += 1

            # And a full pass over the validation data:
            val_err = 0
            val_batches = 0

            for batch in iterate_batches(self.data._X_val, self.data._y_val,
                                         Cfg.batch_size, shuffle=False):

                inputs, _, _ = batch
                err = self.val_network_ocsvm(inputs)
                val_err += err
                val_batches += 1

            # # print results for epoch
            # print("Epoch {} of {} took {:.3f}s".format(
            #     epoch + 1, n_epochs, time.time() - start_time))
            #print("  training loss:\t\t{:.6f}".format(train_err/train_batches))
            #print("  validation loss:\t\t{:.6f}".format(val_err/val_batches))

    def get_learned_features(self):

        pred_train = np.zeros((self.data._X_train.shape[0], self.encoding_size),
                              dtype=np.float32)
        pred_val = np.zeros((self.data._X_val.shape[0], self.encoding_size),
                            dtype=np.float32)

        for batch in iterate_batches(self.data._X_train, self.data._y_train,
                                     Cfg.batch_size, shuffle=False):

            inputs, _, batch_idx = batch
            batch_pred = self.get_network_prediction(inputs)
            start_idx = batch_idx * Cfg.batch_size
            stop_idx = min(len(self.data._X_train), start_idx + Cfg.batch_size)
            pred_train[start_idx:stop_idx, :] = batch_pred

        for batch in iterate_batches(self.data._X_val, self.data._y_val,
                                     Cfg.batch_size, shuffle=False):

            inputs, _, batch_idx = batch
            batch_pred = self.get_network_prediction(inputs)
            start_idx = batch_idx * Cfg.batch_size
            stop_idx = min(len(self.data._X_val), start_idx + Cfg.batch_size)
            pred_val[start_idx:stop_idx, :] = batch_pred

        self.learned_code_train = pred_train
        self.learned_code_val = pred_val

    def normalize_feature_vecs(self):

        na = np.newaxis

        norms_train = np.linalg.norm(self.learned_code_train, axis=1)[:, na]
        norms_val = np.linalg.norm(self.learned_code_val, axis=1)[:, na]

        self.learned_code_train = self.learned_code_train / norms_train
        self.learned_code_val = self.learned_code_val / norms_val

    def initialize_ocsvm(self, kernel='linear'):

        nu = self.nu.get_value()
        self.ocsvm = svm.OneClassSVM(kernel=kernel, nu=nu)

    def train_ocsvm(self):

        self.ocsvm.fit(self.learned_code_train)

        # get one-class SVM parameters
        # score = W*x-rho, where a lower score indicates outliers
        W = self.ocsvm.coef_.swapaxes(0, 1)
        rho = -self.ocsvm.intercept_[0]
        self.ocsvm_W.set_value(W.astype(np.float32))
        self.ocsvm_rho.set_value(rho.astype(np.float32))

        # get auc on train and validation data
        self.ocsvm_scores_train = self.ocsvm.decision_function(
            self.learned_code_train)
        self.ocsvm_auc_train = roc_auc_score(
            self.data._y_train, (-1.0) * self.ocsvm_scores_train.flatten())
        self.ocsvm_scores_val = self.ocsvm.decision_function(
            self.learned_code_val)
        self.ocsvm_auc_val = roc_auc_score(
            self.data._y_val, (-1.0) * self.ocsvm_scores_val.flatten())

        print("Train AUC:\t\t{:.6f}".format(self.ocsvm_auc_train))
        print("Validation AUC:\t\t{:.6f}".format(self.ocsvm_auc_val))
