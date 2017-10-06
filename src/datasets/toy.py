from base import DataLoader
from utils.misc import flush_last_line
from layers.unpool import Unpool2DLayer
from config import Configuration as Cfg

from sklearn.preprocessing import OneHotEncoder
import numpy as np
import lasagne.nonlinearities
import lasagne.layers

class ToySeq_DataLoader(DataLoader):

    def __init__(self):

        DataLoader.__init__(self)

        self.dataset_name = "toyseq"
        self.data_path = ""

        self.n_train = Cfg.toy_n_train
        self.n_val = int(np.ceil(0.25 * Cfg.toy_n_train))
        self.n_test = int(np.ceil(0.25 * Cfg.toy_n_train))

        self.n_classes = 2  # normal and anomalous/outlier

        Cfg.n_batches = int(np.ceil(self.n_train * 1. / Cfg.batch_size))

        self.on_memory = True
        Cfg.store_on_gpu = True

        # generation parameters
        self.seed = Cfg.seed
        self.len_seq = 32

        # generate data
        self.load_data()

        # add motif
        nucs = np.array([0, 1, 2, 3])  # ['A', 'C', 'G', 'T']
        self.motif = np.random.choice(nucs, Cfg.toy_motif_len)
        self.add_motif(self.motif, frac=Cfg.out_frac, offset=Cfg.toy_motif_off)

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
        '''
        Generate genomic sequences of length self.len_seq in one-hot encoding
        with ['A', 'C', 'G', 'T'] encoded by indices [0, 1, 2, 3].
        The shape of the final training and validation set is similar to images:
        (n_samples, 1, 4, len_seq).
        '''

        print("Generating data...")

        nucs = np.array([0, 1, 2, 3])  # ['A', 'C', 'G', 'T']

        # generate data
        np.random.seed(self.seed)
        X_train = np.random.choice(nucs, (self.n_train, self.len_seq))
        X_val = np.random.choice(nucs, (self.n_val, self.len_seq))
        X_test = np.random.choice(nucs, (self.n_test, self.len_seq))

        # one-hot encoding
        enc = OneHotEncoder(n_values=4)

        self._X_train = enc.fit_transform(X_train).toarray()
        self._X_train = np.reshape(self._X_train, (self.n_train, 1,
                                                   self.len_seq, 4))
        self._X_train = np.swapaxes(self._X_train, 2, 3)
        self._y_train = np.zeros(self.n_train)

        self._X_val = enc.fit_transform(X_val).toarray()
        self._X_val = np.reshape(self._X_val, (self.n_val, 1,
                                               self.len_seq, 4))
        self._X_val = np.swapaxes(self._X_val, 2, 3)
        self._y_val = np.zeros(self.n_val)

        self._X_test = enc.fit_transform(X_test).toarray()
        self._X_test = np.reshape(self._X_test, (self.n_test, 1,
                                                 self.len_seq, 4))
        self._X_test = np.swapaxes(self._X_test, 2, 3)
        self._y_test = np.zeros(self.n_test)

        flush_last_line()
        print("Data generated.")

    def add_motif(self, motif, frac=0.1, offset=0):

        len_motif = len(motif)

        idx_out_train = int(np.ceil(self.n_train * (1-frac)))
        idx_out_val = int(np.ceil(self.n_val * (1 - frac)))
        idx_out_test = int(np.ceil(self.n_test * (1 - frac)))

        # reset nucs accordingly
        self._X_train[idx_out_train:, 0, :, offset:(offset + len_motif)] = 0
        self._X_val[idx_out_val:, 0, :, offset:(offset + len_motif)] = 0
        self._X_test[idx_out_test:, 0, :, offset:(offset + len_motif)] = 0

        # add motif
        i = 0
        for nuc in motif:
            self._X_train[idx_out_train:, 0, nuc, offset+i] = 1
            self._X_val[idx_out_val:, 0, nuc, offset + i] = 1
            self._X_test[idx_out_test:, 0, nuc, offset + i] = 1
            i += 1

        self._y_train[idx_out_train:] = 1
        self._y_val[idx_out_val:] = 1
        self._y_test[idx_out_test:] = 1

    def build_architecture(self, nnet):

        nnet.addInputLayer(shape=(None, 1, 4, self.len_seq))

        nnet.addConvLayer(num_filters=128, filter_size=(4, 5), pad=(0, 2))
        nnet.addReLU()
        nnet.addMaxPool(pool_size=(1, 2))
        nnet.addDimshuffleLayer(pattern=(0, 2, 1, 3))  # reorder dimensions

        nnet.addConvLayer(num_filters=128, filter_size=(128, 5), pad=(0, 2))
        nnet.addReLU()
        nnet.addMaxPool(pool_size=(1, 2))
        nnet.addDimshuffleLayer(pattern=(0, 2, 1, 3))  # reorder dimensions

        nnet.addConvLayer(num_filters=128, filter_size=(128, 5), pad=(0, 2))
        nnet.addReLU()
        nnet.addMaxPool(pool_size=(1, 8))

        nnet.addDropoutLayer()
        nnet.addDenseLayer(num_units=32)
        nnet.addReLU()

        nnet.addDropoutLayer()
        nnet.addDenseLayer(num_units=32)
        nnet.addReLU()

        nnet.addDropoutLayer()
        nnet.addDenseLayer(num_units=32)
        nnet.addReLU()

        nnet.addDropoutLayer()
        nnet.addDenseLayer(num_units=32)
        nnet.addReLU()

        if Cfg.ocsvm_loss:
            nnet.addDenseLayer(num_units=32, normalize=True)
            nnet.addNormLayer()
            self.ocsvm_layer = nnet.addOCSVMLayer()
        elif not Cfg.softmax_loss:
            nnet.addDenseLayer(num_units=32)
            nnet.addReLU()
            nnet.addSVMLayer()
        else:
            nnet.addDropoutLayer()
            nnet.addDenseLayer(num_units=32)
            nnet.addReLU()
            nnet.addDenseLayer(num_units=1,
                               nonlinearity=lasagne.nonlinearities.sigmoid)

    def build_autoencoder(self, autoencoder):

        # Input
        encoder = lasagne.layers.InputLayer(shape=(None, 1, 4, self.len_seq))

        # Convolution and max-pool
        encoder = lasagne.layers.Conv2DLayer(
            encoder, num_filters=128, filter_size=(4, 5), stride=(1, 1),
            pad=(0, 2), nonlinearity=lasagne.nonlinearities.rectify)
        encoder = lasagne.layers.Pool2DLayer(
            encoder, pool_size=(1, 2), stride=None, pad=(0, 0),
            ignore_border=True, mode='max')
        encoder = lasagne.layers.DimshuffleLayer(encoder, pattern=(0, 2, 1, 3))

        # Convolution and max-pool
        encoder = lasagne.layers.Conv2DLayer(
            encoder, num_filters=128, filter_size=(128, 5), stride=(1, 1),
            pad=(0, 2), nonlinearity=lasagne.nonlinearities.rectify)
        encoder = lasagne.layers.Pool2DLayer(
            encoder, pool_size=(1, 2), stride=None, pad=(0, 0),
            ignore_border=True, mode='max')
        encoder = lasagne.layers.DimshuffleLayer(encoder, pattern=(0, 2, 1, 3))

        # Convolution and max-pool
        encoder = lasagne.layers.Conv2DLayer(
            encoder, num_filters=128, filter_size=(128, 5), stride=(1, 1),
            pad=(0, 2), nonlinearity=lasagne.nonlinearities.rectify)
        encoder = lasagne.layers.Pool2DLayer(
            encoder, pool_size=(1, 8), stride=None, pad=(0, 0),
            ignore_border=True, mode='max')

        # Dropout and dense layers
        encoder = lasagne.layers.DropoutLayer(encoder, p=0.5, rescale=True)
        encoder = lasagne.layers.DenseLayer(encoder, num_units=64)

        encoder = lasagne.layers.DropoutLayer(encoder, p=0.5, rescale=True)
        encoder = lasagne.layers.DenseLayer(encoder, num_units=32)

        # Encoding
        encoder = lasagne.layers.DenseLayer(encoder,
                                            num_units=autoencoder.encoding_size)

        # Dropout and dense layers
        decoder = lasagne.layers.DenseLayer(encoder, num_units=32)
        decoder = lasagne.layers.DropoutLayer(decoder, p=0.5, rescale=True)

        decoder = lasagne.layers.DenseLayer(decoder, num_units=64)
        decoder = lasagne.layers.DropoutLayer(decoder, p=0.5, rescale=True)

        # Unpooling and deconvolution
        decoder = lasagne.layers.DenseLayer(decoder, num_units=128)
        decoder = lasagne.layers.ReshapeLayer(decoder, shape=([0], 128, 1, 1))
        decoder = Unpool2DLayer(decoder, ds=(1, 8))
        decoder = lasagne.layers.DimshuffleLayer(decoder, pattern=(0, 2, 1, 3))
        decoder = lasagne.layers.Conv2DLayer(
            decoder, num_filters=128, filter_size=(128, 5), stride=(1, 1),
            pad=(0, 2))
        decoder = Unpool2DLayer(decoder, ds=(1, 2))
        decoder = lasagne.layers.DimshuffleLayer(decoder, pattern=(0, 2, 1, 3))
        decoder = lasagne.layers.Conv2DLayer(
            decoder, num_filters=128, filter_size=(128, 5), stride=(1, 1),
            pad=(0, 2))
        decoder = Unpool2DLayer(decoder, ds=(1, 2))
        decoder = lasagne.layers.DimshuffleLayer(decoder, pattern=(0, 2, 1, 3))
        decoder = lasagne.layers.Conv2DLayer(
            decoder, num_filters=4, filter_size=(128, 5), stride=(1, 1),
            pad=(0, 2))

        # Output
        decoder = lasagne.layers.DimshuffleLayer(decoder, pattern=(0, 2, 1, 3))

        autoencoder.encoder = encoder
        autoencoder.decoder = decoder

    def check_specific(self):

        # store primal variables on RAM
        assert Cfg.store_on_gpu
