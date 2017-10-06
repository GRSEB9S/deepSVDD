import argparse
import numpy as np
import cPickle as pickle
import os
import sys
import theano

from neuralnet import NeuralNet
from config import Configuration as Cfg
from utils.visualization.line_plot import plot_line
from utils.visualization.five_number_plot import plot_five_number_summary

# ====================================================================
# Parse arguments
# --------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--device",
                    help="Computation device to use for experiment",
                    type=str, default="cpu")
parser.add_argument("--xp_dir",
                    help="directory for the experiment",
                    type=str)
parser.add_argument("--n_epochs",
                    help="number of epochs",
                    type=int)
parser.add_argument("--batch_size",
                    help="batch size",
                    type=int, default=200)
parser.add_argument("--in_name",
                    help="name for inputs of experiment",
                    type=str, default="")
parser.add_argument("--out_name",
                    help="name for outputs of experiment",
                    type=str, default="")
parser.add_argument("--nu",
                    help="nu parameter in one-class SVM",
                    type=float, default=0.5)
parser.add_argument("--out_frac",
                    help="fraction of outliers in data set",
                    type=float, default=0.1)
parser.add_argument("--seed",
                    help="numpy seed",
                    type=int, default=0)
parser.add_argument("--net_depth",
                    help="depth of the network",
                    type=int, default=2)
parser.add_argument("--net_width",
                    help="width of the network",
                    type=int, default=4)

################################################################################


def main():

    args = parser.parse_args()
    print('Options:')
    for (key, value) in vars(args).iteritems():
        print("{:16}: {}".format(key, value))

    # computation device
    if 'gpu' in args.device:
        theano.sandbox.cuda.use(args.device)

    # update config data
    # dataset
    dataset = 'normal'
    Cfg.seed = args.seed
    Cfg.out_frac = args.out_frac
    Cfg.toy_n_train = 10000
    Cfg.toy_ndim = 2
    Cfg.toy_net_depth = args.net_depth
    Cfg.toy_net_width = args.net_width
    str_architecture = "mlp_" + str(args.net_depth) + "x" + str(args.net_width)
    # neural network parameters
    loss = 'ocsvm'
    Cfg.compile_lwsvm = False
    Cfg.softmax_loss = (loss == 'ce')
    Cfg.ocsvm_loss = (loss == 'ocsvm')
    # optimization parameters
    momentum = 0.9
    Cfg.momentum.set_value(Cfg.floatX(momentum))
    rho = 0.9
    Cfg.rho.set_value(Cfg.floatX(rho))
    n_epochs = args.n_epochs
    Cfg.batch_size = args.batch_size
    # OC-SVM layer parameters
    nu = 0.1
    Cfg.nu.set_value(Cfg.floatX(nu))
    # Regularization
    Cfg.include_bias = True
    Cfg.dropout = False
    Cfg.dropout_architecture = False

    # loop ranges
    solvers = ["sgd", "momentum", "nesterov", "adagrad", "rmsprop", "adadelta",
               "adam", "adamax"]
    Cs = [None] + [float(10 ** j) for j in range(-3, 4)]
    lrs = [float(10 ** j) for j in range(-4, 2)]

    n_loops = len(solvers) * 2 * len(Cs) * len(lrs)
    loop = 0

    for solver in solvers:

        for switch in [False, True]:
            Cfg.normalize = switch

            if switch:
                str_normalization = "with_normalization"
            else:
                str_normalization = "wo_normalization"

            i = 0

            for C in Cs:
                str_C = "C = " + str(C)
                if C is not None:
                    Cfg.weight_decay = True
                    Cfg.C.set_value(Cfg.floatX(C))
                else:
                    Cfg.weight_decay = False

                j = 0

                for lr in lrs:
                    str_lr = "lr = " + str(lr)
                    Cfg.learning_rate.set_value(Cfg.floatX(lr))

                    xp_path = "{}/{}/{}/{}/".format(args.xp_dir,
                                                    str_architecture,
                                                    solver,
                                                    str_normalization)
                    if not os.path.exists(xp_path):
                        os.makedirs(xp_path)

                    save_to = "{}/initial_weights.p".format(xp_path)

                    loop += 1

                    try:
                        print("Starting loop {} of {}".format(str(loop),
                                                           str(n_loops)))
                        print("Train: {}, {}, C={}, lr={}".format(solver,
                                                                  str(switch),
                                                                  str(C),
                                                                  str(lr)))

                        sgdocsvm = NeuralNet(dataset=dataset,
                                             use_weights=None)
                        sgdocsvm.train(solver=solver, n_epochs=n_epochs,
                                       save_to=save_to)

                        # plot
                        suffix = "({}, {}, {})".format(solver, str_C, str_lr)

                        # Train objective (and its parts)
                        objective = {"objective": sgdocsvm.objective_train,
                                     "l2 penalty": sgdocsvm.l2_penalty,
                                     "W_svm": sgdocsvm.Wsvm_norm,
                                     "emp. loss": sgdocsvm.emp_loss_train,
                                     "rho": sgdocsvm.rhosvm}
                        plot_line(objective,
                                  title="Train objective " + suffix,
                                  ylabel="Objective",
                                  export_pdf=(xp_path + "/obj_train_"
                                              + str(i) + str(j)))

                        # Validation objective (and its parts)
                        objective = {"objective": sgdocsvm.objective_val,
                                     "l2 penalty": sgdocsvm.l2_penalty,
                                     "W_svm": sgdocsvm.Wsvm_norm,
                                     "emp. loss": sgdocsvm.emp_loss_val,
                                     "rho": sgdocsvm.rhosvm}
                        plot_line(objective,
                                  title="Test objective " + suffix,
                                  ylabel="Objective",
                                  export_pdf=(xp_path + "/obj_test_"
                                              + str(i) + str(j)))

                        # Train and validation set AUC
                        auc = {"train": sgdocsvm.auc_train,
                               "test": sgdocsvm.auc_val}
                        plot_line(auc, title="AUC " + suffix,
                                  ylabel="AUC",
                                  export_pdf=(xp_path + "/auc_"
                                              + str(i) + str(j)))

                        # Five number summary of scores
                        title = "Summary of train scores " + suffix
                        ylab = "Score"

                        scores = {"normal": sgdocsvm.scores_train[
                            sgdocsvm.data._y_train == 0],
                                  "outlier": sgdocsvm.scores_train[
                                      sgdocsvm.data._y_train == 1]}
                        plot_five_number_summary(scores, title=title,
                                                 ylabel=ylab,
                                                 export_pdf=(xp_path
                                                             + "/scores_train_"
                                                             + str(i) + str(j)))

                        title = "Summary of test scores " + suffix
                        scores = {"normal": sgdocsvm.scores_val[
                            sgdocsvm.data._y_val == 0],
                                  "outlier": sgdocsvm.scores_val[
                                      sgdocsvm.data._y_val == 1]}
                        plot_five_number_summary(scores, title=title,
                                                 ylabel=ylab,
                                                 export_pdf=(xp_path
                                                             + "/scores_test_"
                                                             + str(i) + str(j)))

                        # Five number summary of feature representation norms
                        title = "Summary of train feature rep. norms " + suffix
                        ylab = "Feature representation norm"

                        rep_norm = {"normal": sgdocsvm.rep_norm_train[
                            sgdocsvm.data._y_train == 0],
                                    "outlier": sgdocsvm.rep_norm_train[
                                        sgdocsvm.data._y_train == 1]}
                        plot_five_number_summary(rep_norm, title=title,
                                                 ylabel=ylab,
                                                 export_pdf=(xp_path
                                                             + "/rep_norm_train_"
                                                             + str(i) + str(j)))

                        title = "Summary of test feature rep. norms " + suffix
                        rep_norm = {"normal": sgdocsvm.rep_norm_val[
                            sgdocsvm.data._y_val == 0],
                                    "outlier": sgdocsvm.rep_norm_val[
                                        sgdocsvm.data._y_val == 1]}
                        plot_five_number_summary(rep_norm, title=title,
                                                 ylabel=ylab,
                                                 export_pdf=(xp_path
                                                             + "/rep_norm_test_"
                                                             + str(i) + str(j)))

                        # log
                        sgdocsvm.log.save_to_file(
                            "{}/results_{}{}.p".format(xp_path, str(i), str(j)))
                        sgdocsvm.dump_weights(
                            "{}/final_weights_{}{}.p".format(xp_path,
                                                             str(i), str(j)))

                        log_file = "{}/log_{}{}.txt".format(xp_path,
                                                             str(i), str(j))

                        logged = open(log_file, "a")
                        logged.write("Dataset: {}\n".format(dataset))
                        logged.write("Seed: {}\n".format(Cfg.seed))
                        logged.write("Loss: {}\n".format(loss))
                        logged.write("Solver: {}\n".format(solver))
                        logged.write("Learning rate: {}\n".format(lr))
                        logged.write("Momentum: {}\n".format(momentum))
                        logged.write("Rho: {}\n".format(rho))
                        logged.write("Number of epochs: {}\n".format(n_epochs))
                        logged.write(
                            "Batch size: {}\n\n".format(Cfg.batch_size))

                        logged.write("Regularization\n")
                        logged.write(
                            "Weight decay: {}\n".format(Cfg.weight_decay))
                        logged.write(
                            "Bias penalized? {}\n".format(Cfg.include_bias))
                        logged.write("C-parameter: {}\n".format(C))
                        logged.write("Dropout: {}\n".format(Cfg.dropout))
                        logged.write("Dropout architecture? {}\n\n".format(
                            Cfg.dropout_architecture))

                        logged.write("OC-SVM\n")
                        logged.write("Fraction of Outliers: {}\n".format(
                            Cfg.out_frac))
                        logged.write("Nu-parameter: {}\n".format(nu))
                        logged.write("Normalize feature map? {}\n\n".format(
                            Cfg.normalize))
                        logged.close()

                    except:
                        pass

                    j += 1

                i += 1


if __name__ == '__main__':
    main()
