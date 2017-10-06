from datasets.base import DataLoader
from datasets.preprocessing import crop_to_square, downscale, gcn, \
    learn_dictionary

from utils.visualization.mosaic_plot import plot_mosaic
from utils.misc import flush_last_line
from config import Configuration as Cfg

import os
import lmdb
import cv2
import numpy as np
import cPickle as pickle


class Bedroom_DataLoader(DataLoader):

    def __init__(self):

        DataLoader.__init__(self)

        self.dataset_name = "bedroom"

        self.n_train = 3033042
        self.n_val = 300
        self.n_test = 3000  # test set from val sets w/ 300 samples per category
        # Original test set with 10000 samples has no public labels

        self.seed = Cfg.seed

        Cfg.out_frac = 0
        self.n_classes = 2

        self.test_categories = ['bridge', 'church_outdoor', 'classroom',
                                'conference_room', 'dining_room', 'kitchen',
                                'living_room', 'restaurant', 'tower']

        Cfg.n_batches = Cfg.bedroom_n_train_samples / Cfg.batch_size

        self.data_path = '../data/lsun/'
        self.train_db = 'bedroom_train_lmdb'
        self.val_db = 'bedroom_val_lmdb'

        self.on_memory = False
        Cfg.store_on_gpu = False

        # load data from disk
        self.load_data()

        # prepare online learning over random batches from training set
        np.random.seed(self.seed)
        self.idx_train_perm = np.arange(self.n_train)
        np.random.shuffle(self.idx_train_perm)

    def load_data(self, original_scale=False):

        print("Loading data...")

        # open train db
        env = lmdb.open(self.data_path + self.train_db, max_readers=1,
                        readonly=True, readahead=False, meminit=False)
        with env.begin(write=False) as txn:
            self.n_train = int(txn.stat()['entries'])

        # write train data keys to cache file (batches will iterate over keys)
        cache_file = self.data_path + '_cache_' + self.train_db
        if os.path.isfile(cache_file):
            self.train_keys = pickle.load(open(cache_file, "rb"))
        else:
            with env.begin(write=False) as txn:
                self.train_keys = [key for key, _ in txn.cursor()]
            pickle.dump(self.train_keys, open(cache_file, "wb"))
        env.close()

        # labels always 0 for each sample per batch since no outliers in train
        self._y_train = np.zeros(Cfg.batch_size, dtype=np.int32)

        # open val db and load validation images
        env = lmdb.open(self.data_path + self.val_db, max_readers=1,
                        readonly=True, readahead=False, meminit=False)
        with env.begin(write=False) as txn:
            self.n_val = int(txn.stat()['entries'])
            cursor = txn.cursor()

            X = []
            for key, val in cursor:
                img = cv2.imdecode(np.fromstring(val, dtype=np.uint8), 1)
                img = img[:, :, ::-1]  # OpenCV has BGR, matplotlib RGB order
                # reshape, crop and downsample
                img = np.rollaxis(img, 2)
                img = crop_to_square(img)  # img.shape = (3, 256, 256)
                img = downscale(img, pixels=Cfg.bedroom_downscale_pxl)

                X.append(img)
        env.close()

        self._X_val = np.concatenate(X).reshape(-1, 3, 64, 64).astype(np.float32)
        self._X_val /= np.float32(255)  # simple rescaling to [0,1]
        self._y_val = np.zeros(self.n_val, dtype=np.int32)

        # normalize data (if original scale should not be preserved)
        if not original_scale:

            # global contrast normalization
            if Cfg.gcn:
                gcn(self._X_val, scale=Cfg.unit_norm_used)

            # rescale to [0,1] (w.r.t. min and max in validation data)
            self.min_pxl_val = np.min(self._X_val)
            self.max_pxl_val = np.max(self._X_val)
            self._X_val -= self.min_pxl_val
            self._X_val /= (self.max_pxl_val - self.min_pxl_val)

        # open val databases of other categories to build test set
        for category in self.test_categories:
            env = lmdb.open(self.data_path + category + '_val_lmdb',
                            max_readers=1, readonly=True, readahead=False,
                            meminit=False)
            with env.begin(write=False) as txn:
                cursor = txn.cursor()

                for key, val in cursor:
                    img = cv2.imdecode(np.fromstring(val, dtype=np.uint8), 1)
                    img = img[:,:,::-1]  # OpenCV has BGR, matplotlib RGB order
                    # reshape, crop and downsample
                    img = np.rollaxis(img, 2)
                    img = crop_to_square(img)  # img.shape = (3, 256, 256)
                    img = downscale(img, pixels=Cfg.bedroom_downscale_pxl)

                    X.append(img)  # continued from validation set of bedroom
            env.close()

            self._X_test = np.concatenate(X).reshape(-1, 3, 64, 64).astype(
                np.float32)
            self._X_test /= np.float32(255)  # simple rescaling to [0,1]
            self._y_test = np.ones(self.n_test, dtype=np.int32)
            self._y_test[:self.n_val] = 0

            # normalize data (if original scale should not be preserved)
            if not original_scale:
                # global contrast normalization
                if Cfg.gcn:
                    gcn(self._X_test, scale=Cfg.unit_norm_used)

                # rescale to [0,1] (w.r.t. min and max in validation data)
                self._X_test -= self.min_pxl_val
                self._X_test /= (self.max_pxl_val - self.min_pxl_val)

        flush_last_line()
        print("Data loaded.")


    def load_train_batch_bedroom(self, batch):

        start_idx = batch * Cfg.batch_size
        stop_idx = min(self.n_train, start_idx + Cfg.batch_size)
        batch_idx = self.idx_train_perm[start_idx:stop_idx]

        # open train db
        env = lmdb.open(self.data_path + self.train_db, max_readers=1,
                        readonly=True, readahead=False, meminit=False)
        with env.begin(write=False) as txn:
            cursor = txn.cursor()

            X = []
            for key in [self.train_keys[idx] for idx in batch_idx]:
                val = cursor.get(key)
                img = cv2.imdecode(np.fromstring(val, dtype=np.uint8), 1)
                img = img[:, :, ::-1]  # OpenCV has BGR, matplotlib RGB order
                # reshape, crop and downsample
                img = np.rollaxis(img, 2)
                img = crop_to_square(img)  # img.shape = (3, 256, 256)
                img = downscale(img, pixels=Cfg.bedroom_downscale_pxl)

                X.append(img)

        self._X_train_batch = np.concatenate(X).reshape(-1, 3, 64, 64).astype(
            np.float32)

        # pre-process batch accordingly

        # simple rescaling to [0,1]
        self._X_train_batch /= np.float32(255)

        # global contrast normalization
        if Cfg.gcn:
            gcn(self._X_train_batch, scale=Cfg.unit_norm_used)

        # rescale to [0,1] (w.r.t. min and max in validation data)
        self._X_train_batch -= self.min_pxl_val
        self._X_train_batch /= (self.max_pxl_val - self.min_pxl_val)


    def build_architecture(self, nnet):

        if Cfg.weight_dict_init:
            # draw random train batch
            batch_init = np.random.choice(Cfg.n_batches)
            nnet.data.load_train_batch_bedroom(batch_init)

            # initialize first layer filters by atoms of a dictionary
            W1_init = learn_dictionary(nnet.data._X_train_batch, n_filters=16,
                                       filter_size=5, n_sample=Cfg.batch_size,
                                       n_sample_patches=int(5e5))
            plot_mosaic(W1_init, title="First layer filters initialization",
                        canvas="black",
                        export_pdf=(Cfg.xp_path + "/filters_init"))

        shape = (None, 3, 64, 64)
        nnet.addInputLayer(shape=shape)

        if Cfg.weight_dict_init:
            nnet.addConvLayer(num_filters=16,
                              filter_size=(5, 5),
                              pad=1,
                              flip_filters=False, b=None,
                              W=W1_init)
        else:
            nnet.addConvLayer(num_filters=16,
                              filter_size=(5, 5),
                              pad=1,
                              flip_filters=False, b=None)
        nnet.addLeakyReLU()
        nnet.addConvLayer(num_filters=16,
                          filter_size=(5, 5),
                          pad=1,
                          flip_filters=False, b=None)
        nnet.addLeakyReLU()
        nnet.addMaxPool(pool_size=(2, 2))

        nnet.addConvLayer(num_filters=32,
                          filter_size=(5, 5),
                          pad=1,
                          flip_filters=False, b=None)
        nnet.addLeakyReLU()
        nnet.addConvLayer(num_filters=32,
                          filter_size=(5, 5),
                          pad=1,
                          flip_filters=False, b=None)
        nnet.addLeakyReLU()
        nnet.addMaxPool(pool_size=(2, 2))

        nnet.addConvLayer(num_filters=64,
                          filter_size=(5, 5),
                          pad=1,
                          flip_filters=False, b=None)
        nnet.addLeakyReLU()
        nnet.addConvLayer(num_filters=64,
                          filter_size=(5, 5),
                          pad=1,
                          flip_filters=False, b=None)
        nnet.addLeakyReLU()
        nnet.addMaxPool(pool_size=(2, 2))

        nnet.addDenseLayer(num_units=64, b=None)

    def check_specific(self):
        # nothing to check
        return
