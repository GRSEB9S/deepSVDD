from svm import SVM
from config import Configuration as Cfg
from utils.log import log_exp_config, log_SVM, log_AD_results
from utils.visualization.images_plot import plot_outliers, plot_normals


# dataset
dataset = "mnist"  # "mnist", "toyseq", "normal"
Cfg.seed = 0
Cfg.out_frac = 0
Cfg.ad_experiment = True

# "mnist" parameters (-1 stands for all the rest)
Cfg.mnist_normal = 0
Cfg.mnist_outlier = -1

# "cifar10" parameters (-1 stands for all the rest)
Cfg.cifar10_normalize_mode = "fixed value"  # "per channel", "per pixel"
Cfg.cifar10_normal = 1
Cfg.cifar10_outlier = -1

# SVM parameters
loss = "OneClassSVM"
kernel = "rbf"
# mnist: gamma = (1.0 / 784)
gamma = (1.0 / 784) * (10 ** 0)  # if 'auto', then 1/n is taken
verbose = True
Cfg.svm_nu = 0.01

# Plot parameters
Cfg.xp_path = "../log/ocsvm/" + dataset

# initialize OC-SVM
ocsvm = SVM(loss=loss, dataset=dataset, kernel=kernel, gamma=gamma)

# train or load OC-SVM model
ocsvm.train()  # train model
# ocsvm.load_model(filename=Cfg.xp_path + "/model.p")  # load model

# predict scores
ocsvm.predict(which_set='train')
ocsvm.predict(which_set='val')  # validate model on test set

# plot targets and outliers sorted
n_img = 32
plot_outliers(ocsvm, n_img, Cfg.xp_path)
plot_normals(ocsvm, n_img, Cfg.xp_path)

# pickle/serialize
ocsvm.dump_model(filename=Cfg.xp_path + "/model.p")
ocsvm.log_results(filename=Cfg.xp_path + "/AD_results.p")

# log
log_exp_config(Cfg.xp_path, dataset)
log_SVM(Cfg.xp_path, loss, kernel, gamma, Cfg.svm_nu)
log_AD_results(Cfg.xp_path, ocsvm)
