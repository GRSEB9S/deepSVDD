import argparse
import os
import sys
import theano

from neuralnet import NeuralNet
from config import Configuration as Cfg
from utils.log import log_exp_config, log_NeuralNet, log_AD_results
from utils.visualization.diagnostics_plot import plot_diagnostics, \
    plot_ae_diagnostics
from utils.visualization.filters_plot import plot_filters
from utils.visualization.images_plot import plot_outliers, plot_normals, \
    plot_outliers_bedroom, plot_normals_bedroom


# ====================================================================
# Parse arguments
# --------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--dataset",
                    help="dataset name",
                    type=str, choices=["mnist", "cifar10", "toyseq", "bedroom"])
parser.add_argument("--solver",
                    help="solver", type=str,
                    choices=["sgd", "momentum", "nesterov", "adagrad",
                             "rmsprop", "adadelta", "adam", "adamax"])
parser.add_argument("--loss",
                    help="loss function",
                    type=str, choices=["ce", "svm", "ocsvm", "svdd",
                                       "autoencoder"])
parser.add_argument("--lr",
                    help="initial learning rate",
                    type=float)
parser.add_argument("--lr_decay",
                    help="specify if learning rate should be decayed",
                    type=int, default=0)
parser.add_argument("--lr_decay_after_epoch",
                    help="specify the epoch after learning rate should decay",
                    type=int, default=10)
parser.add_argument("--momentum",
                    help="momentum rate if optimization with momentum",
                    type=float, default=0.9)
parser.add_argument("--fastR",
                    help="specify if fastR heuristic for SVDD should be used",
                    type=int, default=0)
parser.add_argument("--pretrain",
                    help="specify if weights should be pre-trained via autoenc",
                    type=int, default=0)
parser.add_argument("--ae_loss",
                    help="specify the reconstruction loss of the autoencoder",
                    type=str, default="l2")
parser.add_argument("--ae_weight_decay",
                    help="specify if weight decay should be used in pretrain",
                    type=int, default=1)
parser.add_argument("--ae_C",
                    help="regularization hyper-parameter in pretrain",
                    type=float, default=1e3)
parser.add_argument("--ae_sparsity_penalty",
                    help="sparsity penalty in pretrain",
                    type=int, default=0)
parser.add_argument("--ae_sparsity_mode",
                    help="specify the mode of the sparsity penalty in pretrain",
                    type=str, default="mean")
parser.add_argument("--ae_B",
                    help="sparsity hyper-parameter in pretrain",
                    type=float, default=1e3)
parser.add_argument("--ae_sparsity",
                    help="Avg activation to regularize sparsity to in pretrain",
                    type=float, default=0.05)
parser.add_argument("--batch_size",
                    help="batch size",
                    type=int, default=200)
parser.add_argument("--n_epochs",
                    help="number of epochs",
                    type=int)
parser.add_argument("--save_at",
                    help="number of epochs before saving model",
                    type=int, default=0)
parser.add_argument("--device",
                    help="Computation device to use for experiment",
                    type=str, default="cpu")
parser.add_argument("--xp_dir",
                    help="directory for the experiment",
                    type=str)
parser.add_argument("--in_name",
                    help="name for inputs of experiment",
                    type=str, default="")
parser.add_argument("--out_name",
                    help="name for outputs of experiment",
                    type=str, default="")
parser.add_argument("--leaky_relu",
                    help="specify if ReLU layer should be leaky",
                    type=int, default=1)
parser.add_argument("--softplus",
                    help="should final layer have softplus activation",
                    type=int, default=0)
parser.add_argument("--Wsvm_penalty",
                    help="specify if weight decay should be applied on Wsvm",
                    type=int, default=0)
parser.add_argument("--weight_decay",
                    help="specify if weight decay should be used",
                    type=int, default=0)
parser.add_argument("--include_bias",
                    help="specify if bias should be penalized in weight decay",
                    type=int, default=0)
parser.add_argument("--C",
                    help="regularization hyper-parameter",
                    type=float, default=1e3)
parser.add_argument("--sparsity_penalty",
                    help="specify if sparsity penalty should be used",
                    type=int, default=0)
parser.add_argument("--sparsity_mode",
                    help="specify the mode of the sparsity penalty",
                    type=str, default="mean")
parser.add_argument("--B",
                    help="sparsity hyper-parameter",
                    type=float, default=1e3)
parser.add_argument("--sparsity",
                    help="Average activation to regularize sparsity to",
                    type=float, default=0.05)
parser.add_argument("--output_penalty",
                    help="specify if output should be penalized",
                    type=int, default=0)
parser.add_argument("--prod_penalty",
                    help="specify if product penalty should be used",
                    type=int, default=0)
parser.add_argument("--dropout",
                    help="specify if dropout layers should be applied",
                    type=int, default=0)
parser.add_argument("--dropout_arch",
                    help="specify if dropout architecture should be used",
                    type=int, default=0)
parser.add_argument("--rho_fixed",
                    help="specify rho=1 fixed in the OC-SVM objective",
                    type=int, default=0)
parser.add_argument("--c_mean_init",
                    help="specify if center c should be initialized as mean",
                    type=int, default=0)
parser.add_argument("--c_mean_init_n_batches",
                    help="from how many batches should the mean be computed?",
                    type=int, default=-1)  # default=-1 means "all"
parser.add_argument("--hard_margin",
                    help="Train deep SVDD with hard-margin algorithm",
                    type=int, default=0)
parser.add_argument("--gaussian_blob",
                    help="Train deep SVDD hard-margin on Gaussian blobs",
                    type=int, default=0)
parser.add_argument("--early_compression",
                    help="specify if early compression algo should be applied",
                    type=int, default=0)
parser.add_argument("--ec_n_epochs",
                    help="number of epochs for early compression algorithm",
                    type=int, default=10)
parser.add_argument("--nu",
                    help="nu parameter in one-class SVM",
                    type=float, default=0.1)
parser.add_argument("--normalize",
                    help="specify if feature map in oc-svm is normalized",
                    type=int, default=0)
parser.add_argument("--ball_penalty",
                    help="specify if ball penalty is added to objective",
                    type=int, default=0)
parser.add_argument("--A",
                    help="ball penalty regularization hyper-parameter",
                    type=float, default=1)
parser.add_argument("--out_frac",
                    help="fraction of outliers in data set",
                    type=float, default=0.1)
parser.add_argument("--seed",
                    help="numpy seed",
                    type=int, default=0)
parser.add_argument("--ad_experiment",
                    help="specify if experiment should be two- or multiclass",
                    type=int, default=1)
parser.add_argument("--weight_dict_init",
                    help="initialize first layer filters by dictionary",
                    type=int, default=0)
parser.add_argument("--pca",
                    help="apply pca in preprocessing",
                    type=int, default=0)
parser.add_argument("--unit_norm",
                    help="specify if data should be scaled to unit norm",
                    type=int, default=1)
parser.add_argument("--unit_norm_used",
                    help="norm to use for scaling the data to unit norm",
                    type=str, default="l2")
parser.add_argument("--z_normalization",
                    help="specify if data should be normalized to Z-Score",
                    type=int, default=0)
parser.add_argument("--gcn",
                    help="apply global contrast normalization in preprocessing",
                    type=int, default=0)
parser.add_argument("--zca_whitening",
                    help="specify if data should be whitened",
                    type=int, default=0)
parser.add_argument("--mnist_bias",
                    help="specify if bias terms are used in MNIST network",
                    type=int, default=1)
parser.add_argument("--mnist_rep_dim",
                    help="specify the dimensionality of the last layer",
                    type=int, default=16)
parser.add_argument("--mnist_normal",
                    help="specify normal class in MNIST",
                    type=int, default=0)
parser.add_argument("--mnist_outlier",
                    help="specify outlier class in MNIST",
                    type=int, default=1)
parser.add_argument("--cifar10_bias",
                    help="specify if bias terms are used in CIFAR-10 network",
                    type=int, default=1)
parser.add_argument("--cifar10_rep_dim",
                    help="specify the dimensionality of the last layer",
                    type=int, default=32)
parser.add_argument("--cifar10_normal",
                    help="specify normal class in CIFAR-10",
                    type=int, default=0)
parser.add_argument("--cifar10_outlier",
                    help="specify outlier class in CIFAR-10",
                    type=int, default=1)
parser.add_argument("--bedroom_n_train",
                    help="specify the number of training samples",
                    type=int, default=1e6)
parser.add_argument("--bedroom_monitor_int",
                    help="specify after how many train samples to monitor",
                    type=int, default=5000)
parser.add_argument("--bedroom_downscale_pxl",
                    help="specify size of images after downscaling",
                    type=int, default=64)

# ====================================================================


def main():

    args = parser.parse_args()
    print('Options:')
    for (key, value) in vars(args).iteritems():
        print("{:16}: {}".format(key, value))

    assert os.path.exists(args.xp_dir)

    # default value for basefile: string basis for all exported file names
    if args.out_name:
        base_file = "{}/{}".format(args.xp_dir, args.out_name)
    else:
        base_file = "{}/{}_{}_{}".format(args.xp_dir, args.dataset,
                                         args.solver, args.loss)

    # if pickle file already there, consider run already done
    if (os.path.exists("{}_weights.p".format(base_file)) and
        os.path.exists("{}_results.p".format(base_file))):
        sys.exit()

    # computation device
    if 'gpu' in args.device:
        theano.sandbox.cuda.use(args.device)

    # set save_at to n_epochs if not provided
    save_at = args.n_epochs if not args.save_at else args.save_at

    save_to = "{}_weights.p".format(base_file)
    weights = "../log/{}_weights.p".format(args.in_name) \
        if args.in_name else None

    # update config data

    # plot parameters
    Cfg.xp_path = args.xp_dir

    # dataset
    Cfg.seed = args.seed
    Cfg.out_frac = args.out_frac
    Cfg.ad_experiment = bool(args.ad_experiment)
    Cfg.weight_dict_init = bool(args.weight_dict_init)
    Cfg.pca = bool(args.pca)
    Cfg.unit_norm = bool(args.unit_norm)
    Cfg.unit_norm_used = args.unit_norm_used
    Cfg.z_normalization = bool(args.z_normalization)
    Cfg.gcn = bool(args.gcn)
    Cfg.zca_whitening = bool(args.zca_whitening)
    Cfg.mnist_bias = bool(args.mnist_bias)
    Cfg.mnist_rep_dim = args.mnist_rep_dim
    Cfg.mnist_normal = args.mnist_normal
    Cfg.mnist_outlier = args.mnist_outlier
    Cfg.cifar10_bias = bool(args.cifar10_bias)
    Cfg.cifar10_rep_dim = args.cifar10_rep_dim
    Cfg.cifar10_normal = args.cifar10_normal
    Cfg.cifar10_outlier = args.cifar10_outlier
    Cfg.bedroom_n_train_samples = args.bedroom_n_train
    Cfg.bedroom_monitor_interval = args.bedroom_monitor_int
    Cfg.bedroom_downscale_pxl = args.bedroom_downscale_pxl

    # neural network
    Cfg.softmax_loss = (args.loss == 'ce')
    Cfg.ocsvm_loss = (args.loss == 'ocsvm')
    Cfg.svdd_loss = (args.loss == 'svdd')
    Cfg.reconstruction_loss = (args.loss == 'autoencoder')
    Cfg.svm_loss = not (Cfg.softmax_loss | Cfg.ocsvm_loss | Cfg.svdd_loss |
                        Cfg.reconstruction_loss)
    Cfg.fastR = bool(args.fastR)
    Cfg.learning_rate.set_value(args.lr)
    Cfg.lr_decay = bool(args.lr_decay)
    Cfg.lr_decay_after_epoch = args.lr_decay_after_epoch
    Cfg.momentum.set_value(args.momentum)
    if args.solver == "rmsprop":
        Cfg.rho.set_value(0.9)
    if args.solver == "adadelta":
        Cfg.rho.set_value(0.95)
    Cfg.batch_size = args.batch_size
    Cfg.compile_lwsvm = False
    Cfg.leaky_relu = bool(args.leaky_relu)
    Cfg.softplus = bool(args.softplus)

    # Pre-training and autoencoder configuration
    Cfg.pretrain = bool(args.pretrain)
    Cfg.ae_loss = args.ae_loss
    Cfg.ae_weight_decay = bool(args.ae_weight_decay)
    Cfg.ae_C.set_value(args.ae_C)
    Cfg.ae_sparsity_penalty = bool(args.ae_sparsity_penalty)
    Cfg.ae_B.set_value(args.ae_B)
    Cfg.ae_sparsity_mode = args.ae_sparsity_mode
    Cfg.ae_sparsity.set_value(args.ae_sparsity)

    # SVDD parameters
    Cfg.c_mean_init = bool(args.c_mean_init)
    if args.c_mean_init_n_batches == -1:
        Cfg.c_mean_init_n_batches = "all"
    else:
        Cfg.c_mean_init_n_batches = args.c_mean_init_n_batches
    Cfg.hard_margin = bool(args.hard_margin)
    Cfg.gaussian_blob = bool(args.gaussian_blob)
    Cfg.early_compression = bool(args.early_compression)
    Cfg.ec_n_epochs = args.ec_n_epochs

    # OC-SVM parameters
    Cfg.rho_fixed = bool(args.rho_fixed)
    Cfg.nu.set_value(args.nu)
    Cfg.normalize = bool(args.normalize)

    # regularization
    Cfg.weight_decay = bool(args.weight_decay)
    Cfg.pow = 2
    Cfg.C.set_value(args.C)
    Cfg.Wsvm_penalty = bool(args.Wsvm_penalty)
    Cfg.sparsity_penalty = bool(args.sparsity_penalty)
    Cfg.sparsity_mode = args.sparsity_mode
    Cfg.B.set_value(args.B)
    Cfg.sparsity.set_value(args.sparsity)
    Cfg.include_bias = bool(args.include_bias)
    Cfg.prod_penalty = bool(args.prod_penalty)
    Cfg.A.set_value(args.A)
    Cfg.ball_penalty = bool(args.ball_penalty)
    Cfg.output_penalty = bool(args.output_penalty)
    Cfg.dropout = bool(args.dropout)
    Cfg.dropout_architecture = bool(args.dropout_arch)

    # train
    nnet = NeuralNet(dataset=args.dataset, use_weights=weights,
                     pretrain=Cfg.pretrain)
    # pre-train weights via autoencoder, if specified
    if Cfg.pretrain:
        nnet.pretrain(solver="adadelta", lr=1.0, n_epochs=100)

    nnet.train(solver=args.solver, n_epochs=args.n_epochs,
               save_at=save_at, save_to=save_to)

    # pickle/serialize AD results
    if Cfg.ad_experiment:
        nnet.log_results(filename=Cfg.xp_path + "/AD_results.p")

    # text log
    nnet.log.save_to_file("{}_results.p".format(base_file))  # save log
    log_exp_config(Cfg.xp_path, args.dataset)
    log_NeuralNet(Cfg.xp_path, args.loss, args.solver, args.lr, args.momentum,
                  None, args.n_epochs, args.C, args.B, args.A, args.nu,
                  args.sparsity)
    if Cfg.ad_experiment:
        log_AD_results(Cfg.xp_path, nnet)

    # plot diagnostics
    # common suffix for plot titles
    str_lr = "lr = " + str(args.lr)
    C = int(args.C)
    if not Cfg.weight_decay:
        C = None
    str_C = "C = " + str(C)
    Cfg.title_suffix = "(" + args.solver + ", " + str_C + ", " + str_lr + ")"

    if args.dataset == "bedroom":
        xlabel = ("Monitoring units (per " +
                  str(Cfg.bedroom_monitor_interval / Cfg.batch_size) +
                  " train batches)")
        plot_diagnostics(nnet, Cfg.xp_path, Cfg.title_suffix, xlabel=xlabel)
    elif args.loss == 'autoencoder':
        plot_ae_diagnostics(nnet, Cfg.xp_path, Cfg.title_suffix)
    else:
        plot_diagnostics(nnet, Cfg.xp_path, Cfg.title_suffix)

    plot_filters(nnet, Cfg.xp_path, Cfg.title_suffix)

    # If AD experiment, plot most anomalous and most normal
    if Cfg.ad_experiment:
        n_img = 32
        if args.dataset == "bedroom":
            plot_outliers_bedroom(nnet, n_img, Cfg.xp_path)
            plot_normals_bedroom(nnet, n_img, Cfg.xp_path)
        else:
            plot_outliers(nnet, n_img, Cfg.xp_path)
            plot_normals(nnet, n_img, Cfg.xp_path)


if __name__ == '__main__':
    main()
