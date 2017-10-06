from kde import KDE
from config import Configuration as Cfg
from utils.log import log_exp_config, log_KDE, log_AD_results
from utils.visualization.images_plot import plot_outliers, plot_normals


# dataset
dataset = "mnist"  # "mnist", "toyseq", "normal"
Cfg.seed = 0
Cfg.out_frac = 0
Cfg.ad_experiment = True
Cfg.pca = True

# "mnist" parameters (-1 stands for all the rest)
Cfg.mnist_normal = 0
Cfg.mnist_outlier = -1

# "cifar10" parameters (-1 stands for all the rest)
Cfg.cifar10_normal = 1
Cfg.cifar10_outlier = -1

kernel = "gaussian"  # gaussian, tophat, epanechnikov, exponential, linear, cosine

# Plot parameters
Cfg.xp_path = "../log/kde/" + dataset

# initialize KDE
kde = KDE(dataset=dataset, kernel=kernel)

# train or load KDE model
kde.train(bandwidth_GridSearchCV=True)  # train model
# kde.load_model(filename=Cfg.xp_path + "/model.p")  # load model

# predict scores
kde.predict(which_set='train')
kde.predict(which_set='val')  # validate model on test set

# plot targets and outliers sorted
n_img = 32
plot_outliers(kde, n_img, Cfg.xp_path)
plot_normals(kde, n_img, Cfg.xp_path)

# pickle/serialize
kde.dump_model(filename=Cfg.xp_path + "/model.p")
kde.log_results(filename=Cfg.xp_path + "/AD_results.p")

# log
log_exp_config(Cfg.xp_path, dataset)
log_KDE(Cfg.xp_path, kernel, bandwidth)
log_AD_results(Cfg.xp_path, kde)
