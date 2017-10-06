import argparse
import numpy as np
import os
import sys
import cPickle as pickle

from neuralnet import NeuralNet
from svm import SVM
from autoencoder import AutoencoderModel
from config import Configuration as Cfg


# Experiment checks:
#
# 1.    Supervised methods should produce high AUCs
# 2.    Can a supervised CNNs compete or exceed SVMs (to know CNN
#       representations work)?
# 3.    Can unsupervised methods OC-SVM w/ WDK detect anomalies correctly?
# 4.    Are unsupervised OC-SVM w/ CNN an improvement?

# ==============================================================================
# Parse arguments
# ------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--dataset",
                    help="dataset name",
                    type=str, choices=["mnist", "cifar10", "cifar100",
                                       "toyseq"])
parser.add_argument("--n_train",
                    help="size of training set",
                    type=int, default=1000)
parser.add_argument("--len_motif",
                    help="length of outlier motif",
                    type=int, default=4)
parser.add_argument("--off_motif",
                    help="offset of outlier motif",
                    type=int, default=0)
parser.add_argument("--nu",
                    help="nu parameter in one-class SVM",
                    type=float, default=0)
parser.add_argument("--xp_dir",
                    help="directory for the experiment",
                    type=str)
parser.add_argument("--seed",
                    help="numpy seed",
                    type=int, default=0)

# ==============================================================================

def main():

    args = parser.parse_args()
    print('Options:')
    for (key, value) in vars(args).iteritems():
        print("{:12}: {}".format(key, value))

    assert os.path.exists(args.xp_dir)

    # default value for basefile: string basis for all exported file names
    base_file = "{}/{}".format(args.xp_dir, args.dataset)

    # if pickle file already there, consider run already done
    if (os.path.exists("{}_weights.p".format(base_file)) and
        os.path.exists("{}_results.p".format(base_file))):
        sys.exit()

    log_file = "{}/log_{}.txt".format(args.xp_dir, args.dataset)

    # set configuration

    # toy data set
    n_sim = 10
    Cfg.toy_seed = args.seed
    Cfg.toy_n_train = args.n_train
    Cfg.toy_motif_len = args.len_motif
    Cfg.toy_motif_off = args.off_motif

    # neural network parameters
    solver = "adagrad"  # "adagrad", "adadelta", "adam"
    Cfg.learning_rate.set_value(Cfg.floatX(0.001))
    n_epochs = int(200)
    Cfg.batch_size = int(100)
    Cfg.C.set_value(Cfg.floatX(100.0))  # l2-regularization parameter
    Cfg.nu.set_value(Cfg.floatX(args.nu))  # if 0 set as fraction of outliers

    # Supervised SVM
    Cfg.svm_C = 1.0

    # One-class parameters
    Cfg.svm_nu = args.nu  # if 0 (default) set as fraction of outliers

    # Autoencoder
    n_iter = 100

    # Fractions of outliers to loop over
    out_frac = np.array([0.01, 0.05, 0.1, 0.15, 0.2])

    # Container for results
    results = [None] * n_sim
    models = {'CNN (supervised)':None,
              'SVM w/ WD (supervised)':None,
              'OC-SVM w/ CNN (Autoencoder)':None,
              'OC-SVM w/ WD':None}

    max_epochs = np.max([n_epochs, n_iter])

    AUC_train = np.zeros((len(models), len(out_frac), max_epochs, n_sim))
    AUC_val = np.zeros((len(models), len(out_frac), max_epochs, n_sim))

    for n in range(n_sim):

        Cfg.toy_seed = n

        l1 = [None] * len(out_frac)
        l2 = [None] * len(out_frac)
        l3 = [None] * len(out_frac)
        l4 = [None] * len(out_frac)

        for i in range(len(out_frac)):

            # set outlier fraction configuration
            Cfg.out_frac = out_frac[i]

            if args.nu == 0:
                Cfg.nu.set_value(Cfg.floatX(out_frac[i]))
                Cfg.svm_nu = out_frac[i]

            # Training

            # CNN (supervised)
            Cfg.softmax_loss = True
            Cfg.ocsvm_loss = False
            supcnn = NeuralNet(dataset=args.dataset, use_weights=None)
            supcnn.train(solver=solver, n_epochs=n_epochs)
            AUC_train[0, i, :n_epochs, n] = supcnn.auc_train.flatten()
            AUC_val[0, i, :n_epochs, n] = supcnn.auc_val.flatten()
            supcnn.flush_data()
            l1[i] = supcnn

            # SVM w/ WD (supervised)
            supsvm = SVM(loss='SVC', dataset=args.dataset)
            supsvm.train(kernel='WeightedDegreeKernel', degree=4,
                         weights=(1, 0.5, 0.25, 0.125))
            AUC_train[1, i, 0, n] = supsvm.auc_train
            AUC_val[1, i, 0, n] = supsvm.auc_val
            supsvm.flush_data()
            l2[i] = supsvm

            # OC-SVM w/ CNN (Autoencoder)
            autoocsvm = AutoencoderModel(dataset=args.dataset)
            autoocsvm.train(n_iter)
            print(autoocsvm.auc_train[-1])
            print(autoocsvm.auc_val[-1])
            AUC_train[2, i, :n_iter, n] = autoocsvm.auc_train.flatten()
            AUC_val[2, i, :n_iter, n] = autoocsvm.auc_val.flatten()
            autoocsvm.flush_data()
            l3[i] = autoocsvm

            # OC-SVM w/ WD
            ocsvm = SVM(loss='OneClassSVM', dataset=args.dataset)
            ocsvm.train(kernel='WeightedDegreeKernel', degree=4,
                        weights=(1, 0.5, 0.25, 0.125))
            AUC_train[3, i, 0, n] = ocsvm.auc_train
            AUC_val[3, i, 0, n] = ocsvm.auc_val
            ocsvm.flush_data()
            l4[i] = ocsvm

        models['CNN (supervised)'] = l1
        models['SVM w/ WD (supervised)'] = l2
        models['OC-SVM w/ CNN (Autoencoder)'] = l3
        models['OC-SVM w/ WD'] = l4

        results[n] = models

    # pickle/serialize
    output = open(args.xp_dir + "/AUC.p", 'wb')
    pickle.dump([AUC_train, AUC_val], output)
    output.close()

    output = open(args.xp_dir + "/results.p", 'wb')
    pickle.dump(results, output)
    output.close()

    logged = open(log_file, "a")
    logged.write("{}: OK\n".format(args.dataset))
    logged.close()


if __name__ == '__main__':
    main()
