import argparse
import os

from svm import SVM
from config import Configuration as Cfg
from utils.log import log_exp_config, log_SVM, log_AD_results
from utils.visualization.images_plot import plot_outliers, plot_normals


# ====================================================================
# Parse arguments
# --------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--dataset",
                    help="dataset name",
                    type=str, choices=["mnist", "cifar10", "toyseq"])
parser.add_argument("--xp_dir",
                    help="directory for the experiment",
                    type=str)
parser.add_argument("--loss",
                    help="loss function",
                    type=str, choices=["OneClassSVM", "SVC"])
parser.add_argument("--kernel",
                    help="kernel",
                    type=str, choices=["linear", "poly", "rbf", "sigmoid",
                                       "DegreeKernel", "WeightedDegreeKernel"])
parser.add_argument("--nu",
                    help="nu parameter in one-class SVM",
                    type=float, default=0.5)
parser.add_argument("--out_frac",
                    help="fraction of outliers in data set",
                    type=float, default=0.1)
parser.add_argument("--seed",
                    help="numpy seed",
                    type=int, default=0)
parser.add_argument("--ad_experiment",
                    help="specify if experiment should be two- or multiclass",
                    type=int, default=1)
parser.add_argument("--unit_norm_used",
                    help="norm to use for scaling the data to unit norm",
                    type=str, default="l2")
parser.add_argument("--gcn",
                    help="apply global contrast normalization in preprocessing",
                    type=int, default=0)
parser.add_argument("--pca",
                    help="apply pca in preprocessing",
                    type=int, default=0)
parser.add_argument("--mnist_normal",
                    help="specify normal class in MNIST",
                    type=int, default=0)
parser.add_argument("--mnist_outlier",
                    help="specify outlier class in MNIST",
                    type=int, default=-1)
parser.add_argument("--cifar10_normal",
                    help="specify normal class in CIFAR-10",
                    type=int, default=1)
parser.add_argument("--cifar10_outlier",
                    help="specify outlier class in CIFAR-10",
                    type=int, default=-1)

# ====================================================================


def main():

    args = parser.parse_args()
    print('Options:')
    for (key, value) in vars(args).iteritems():
        print("{:16}: {}".format(key, value))

    assert os.path.exists(args.xp_dir)

    # update config data

    # plot parameters
    Cfg.xp_path = args.xp_dir

    # dataset
    Cfg.seed = args.seed
    Cfg.out_frac = args.out_frac
    Cfg.ad_experiment = bool(args.ad_experiment)
    Cfg.unit_norm_used = args.unit_norm_used
    Cfg.gcn = bool(args.gcn)
    Cfg.pca = bool(args.pca)
    Cfg.mnist_normal = args.mnist_normal
    Cfg.mnist_outlier = args.mnist_outlier
    Cfg.cifar10_normal = args.cifar10_normal
    Cfg.cifar10_outlier = args.cifar10_outlier

    # SVM parameters
    Cfg.svm_nu = args.nu

    # initialize OC-SVM
    ocsvm = SVM(loss=args.loss, dataset=args.dataset, kernel=args.kernel)

    # train OC-SVM model
    ocsvm.train()

    # predict scores
    ocsvm.predict(which_set='train')
    ocsvm.predict(which_set='val')

    # log
    log_exp_config(Cfg.xp_path, args.dataset)
    log_SVM(Cfg.xp_path, args.loss, args.kernel, ocsvm.svm.gamma, Cfg.svm_nu)
    log_AD_results(Cfg.xp_path, ocsvm)

    # pickle/serialize
    ocsvm.dump_model(filename=Cfg.xp_path + "/model.p")
    ocsvm.log_results(filename=Cfg.xp_path + "/AD_results.p")

    # plot targets and outliers sorted
    n_img = 32
    plot_outliers(ocsvm, n_img, Cfg.xp_path)
    plot_normals(ocsvm, n_img, Cfg.xp_path)


if __name__ == '__main__':
    main()
