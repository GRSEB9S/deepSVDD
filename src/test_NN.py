import os

from neuralnet import NeuralNet
from config import Configuration as Cfg
from utils.log import log_exp_config, log_NeuralNet, log_AD_results
from utils.visualization.diagnostics_plot import plot_diagnostics, \
    plot_ae_diagnostics
from utils.visualization.filters_plot import plot_filters
from utils.visualization.images_plot import plot_outliers, plot_normals
from utils.visualization.scatter_plot import plot_2Dscatter


# dataset
dataset = "mnist"  # "mnist", "toyseq", "normal", "bedroom", "adult"
Cfg.seed = 0
Cfg.out_frac = 0
Cfg.ad_experiment = False
Cfg.weight_dict_init = False
Cfg.pca = False
Cfg.unit_norm = True
Cfg.unit_norm_used = "l1"
Cfg.z_normalization = False
Cfg.gcn = True
Cfg.zca_whitening = False

# "mnist" parameters (-1 stands for all the rest)
Cfg.mnist_bias = False
Cfg.mnist_rep_dim = 16
Cfg.mnist_normal = 0
Cfg.mnist_outlier = -1

# "cifar10" parameters (-1 stands for all the rest)
Cfg.cifar10_bias = False
Cfg.cifar10_rep_dim = 64
Cfg.cifar10_normal = 1
Cfg.cifar10_outlier = -1

# "toyseq" and "normal" parameters
Cfg.toy_n_train = 1000
Cfg.toy_ndim = 2
Cfg.toy_net_depth = 2
Cfg.toy_net_width = 4
Cfg.toy_motif_len = 4
Cfg.toy_motif_off = 10
Cfg.plot_rep = False

# neural network parameters
loss = "svdd"  # "ce", "svm", "ocsvm", "svdd", "autoencoder"

# optimization parameters
# solvers with default parameters:
# "sgd": none
# "momentum": momentum=0.9
# "nesterov": momentum=0.9
# "adagrad": learning_rate=1.0
# "rmsprop": learning_rate=1.0, rho=0.9
# "adadelta": learning_rate=1.0, rho=0.95
# "adam": learning_rate=0.001
# "adamax": learning_rate=0.002
solver = "adam"
Cfg.fastR = False
learning_rate = 0.001
Cfg.lr_decay = False
Cfg.lr_decay_after_epoch = 10
momentum = 0.9
rho = 0.9
n_epochs = 5
Cfg.batch_size = 200

# Pre-Training
Cfg.pretrain = False
Cfg.ae_loss = "l2"
Cfg.ae_diagnostics = True
Cfg.ae_weight_decay = True
Cfg.ae_C.set_value(Cfg.floatX(1000))
Cfg.ae_sparsity_penalty = False
Cfg.ae_B.set_value(Cfg.floatX(1000))
Cfg.ae_sparsity_mode = "mean"
Cfg.ae_sparsity.set_value(Cfg.floatX(0.05))

# Network architecture
Cfg.leaky_relu = True

# SVDD parameters
Cfg.c_mean_init = False
Cfg.c_mean_init_n_batches = "all"
Cfg.hard_margin = True
Cfg.early_compression = False
Cfg.ec_n_epochs = 10

# OC-SVM layer parameters
Cfg.rho_fixed = False
nu = 0.01
Cfg.normalize = False

# Regularization
Cfg.weight_decay = True
C = 1e3
Cfg.sparsity_penalty = False
Cfg.sparsity_mode = "l1"  # "mean" or "l1"
B = 1e5
sparsity = 0.05
Cfg.pow = 2
Cfg.Wsvm_penalty = False
Cfg.include_bias = False
Cfg.spec_penalty = False
Cfg.prod_penalty = False

A = 1.0
Cfg.ball_penalty = False
Cfg.bias_offset = False

Cfg.dropout = False
Cfg.dropout_architecture = False

# Should weights be loaded?
Cfg.xp_path = "../log/" + dataset
in_name = ""  # in_name = "0_mnist_adagrad_svm_weights.p"
weights = "../log/{}".format(in_name) if in_name else None

# initialize neural network
Cfg.learning_rate.set_value(Cfg.floatX(learning_rate))
Cfg.momentum.set_value(Cfg.floatX(momentum))
Cfg.rho.set_value(Cfg.floatX(rho))
Cfg.nu.set_value(Cfg.floatX(nu))
Cfg.C.set_value(Cfg.floatX(C))
Cfg.B.set_value(Cfg.floatX(B))
Cfg.sparsity.set_value(Cfg.floatX(sparsity))
Cfg.A.set_value(Cfg.floatX(A))

Cfg.compile_lwsvm = False
Cfg.softmax_loss = (loss == 'ce')
Cfg.ocsvm_loss = (loss == 'ocsvm')
Cfg.svdd_loss = (loss == 'svdd')
Cfg.reconstruction_loss = (loss == 'autoencoder')
Cfg.svm_loss = not (Cfg.softmax_loss | Cfg.ocsvm_loss | Cfg.svdd_loss |
                    Cfg.reconstruction_loss)

nnet = NeuralNet(dataset=dataset, use_weights=weights, pretrain=Cfg.pretrain)

# plot normal dataset
if dataset == "normal":
    if not os.path.exists(Cfg.xp_path):
        os.makedirs(Cfg.xp_path)
    if not os.path.exists(Cfg.xp_path + "/data"):
        os.makedirs(Cfg.xp_path + "/data")

    title = "Train dataset: " + dataset + " (seed: " + str(Cfg.seed) + ")"
    data = {"normal":nnet.data._X_train[nnet.data._y_train == 0],
            "outlier": nnet.data._X_train[nnet.data._y_train == 1]}
    plot_2Dscatter(data, title=title, export_pdf=(Cfg.xp_path + "/data/train_rep0"))
    title = "Test dataset: " + dataset + " (seed: " + str(Cfg.seed) + ")"
    data = {"normal":nnet.data._X_val[nnet.data._y_val == 0],
            "outlier": nnet.data._X_val[nnet.data._y_val == 1]}
    plot_2Dscatter(data, title=title, export_pdf=(Cfg.xp_path + "/data/test_rep0"))

# pretrain
if Cfg.pretrain:
    nnet.pretrain(solver="adadelta", lr=1.0, n_epochs=2)

# train neural network
nnet.train(solver=solver, n_epochs=n_epochs)

# pickle/serialize AD results
# nnet.log_results(filename=Cfg.xp_path + "/AD_results.p")

# plot diagnostics
# common suffix for plot titles
str_lr = "lr = " + str(learning_rate)
C = int(C)
if not Cfg.weight_decay:
    C = None
str_C = "C = " + str(C)
Cfg.title_suffix = "(" + solver + ", " + str_C + ", " + str_lr + ")"

if loss == 'autoencoder':
    plot_ae_diagnostics(nnet, Cfg.xp_path, Cfg.title_suffix)
else:
    plot_diagnostics(nnet, Cfg.xp_path, Cfg.title_suffix)

plot_filters(nnet, Cfg.xp_path, Cfg.title_suffix)

n_img = 32
plot_outliers(nnet, n_img, Cfg.xp_path)
plot_normals(nnet, n_img, Cfg.xp_path)

# log
log_exp_config(Cfg.xp_path, dataset)
log_NeuralNet(Cfg.xp_path, loss, solver, learning_rate, momentum, rho, n_epochs,
              C, B, A, nu, sparsity)

log_AD_results(Cfg.xp_path, nnet)
