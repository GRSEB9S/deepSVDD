from collections import OrderedDict
from config import Configuration as Cfg
from utils.visualization.line_plot import plot_line
from utils.visualization.five_number_plot import plot_five_number_summary


def plot_diagnostics(nnet, xp_path, title_suffix, xlabel="Epochs",
                     file_prefix=""):
    """
    a function to wrap different diagnostic plots
    """

    # plot train and validation/test objective
    plot_objective(nnet, xp_path, title_suffix, xlabel, file_prefix)

    # plot accuracy
    plot_accuracy(nnet, xp_path, title_suffix, xlabel, file_prefix)

    # plot norms of network parameters (and parameter updates)
    plot_parameter_norms(nnet, xp_path, title_suffix, xlabel, file_prefix)

    if nnet.data.n_classes == 2:
        # plot auc
        plot_auc(nnet, xp_path, title_suffix, xlabel, file_prefix)

        # plot scores
        plot_scores(nnet, xp_path, title_suffix, xlabel, file_prefix)

        # plot norms of feature representations
        plot_representation_norms(nnet, xp_path, title_suffix, xlabel,
                                  file_prefix)


def plot_ae_diagnostics(nnet, xp_path, title_suffix):

    xlabel = "Epochs"
    file_prefix = "ae_"

    # plot train and validation/test objective
    plot_objective(nnet, xp_path, title_suffix, xlabel, file_prefix,
                   pretrain=True)

    # plot auc
    plot_auc(nnet, xp_path, title_suffix, xlabel, file_prefix)

    # plot scores
    plot_scores(nnet, xp_path, title_suffix, xlabel, file_prefix)


def plot_objective(nnet, xp_path, title_suffix, xlabel, file_prefix,
                   pretrain=False):
    """
    plot train and validation/test objective.
    """

    # Plot train objective (and its parts)
    objective = OrderedDict([("objective", nnet.objective_train),
                             ("emp. loss", nnet.emp_loss_train),
                             ("l2 penalty", nnet.l2_penalty)])
    if Cfg.ocsvm_loss:
        objective["W_svm"] = nnet.Wsvm_norm
        objective["rho"] = nnet.rhosvm
    if Cfg.sparsity_penalty:
        objective["sparsity penalty"] = nnet.sparsity_penalty_train
    if Cfg.ae_sparsity_penalty and pretrain:
        objective["sparsity penalty"] = nnet.sparsity_penalty_train
    if Cfg.ball_penalty:
        objective["ball penalty"] = nnet.ball_penalty_train
    if Cfg.output_penalty:
        objective["output penalty"] = nnet.output_penalty_train
    if Cfg.svdd_loss and not pretrain:
        objective["R"] = nnet.R

    # hacky extraction of monitoring values for bedroom dataset
    if nnet.data.dataset_name == "bedroom":
        title = "Validation objective " + title_suffix
    else:
        title = "Train objective " + title_suffix

    plot_line(objective,
              title=title,
              xlabel=xlabel,
              ylabel="Objective",
              export_pdf=(xp_path + "/" + file_prefix + "obj_train"))

    # Plot validation/test objective (and its parts)
    objective = OrderedDict([("objective", nnet.objective_val),
                             ("emp. loss", nnet.emp_loss_val),
                             ("l2 penalty", nnet.l2_penalty)])
    if Cfg.ocsvm_loss:
        objective["W_svm"] = nnet.Wsvm_norm
        objective["rho"] = nnet.rhosvm
    if Cfg.sparsity_penalty:
        objective["sparsity penalty"] = nnet.sparsity_penalty_val
    if Cfg.ae_sparsity_penalty and pretrain:
        objective["sparsity penalty"] = nnet.sparsity_penalty_val
    if Cfg.ball_penalty:
        objective["ball penalty"] = nnet.ball_penalty_val
    if Cfg.output_penalty:
        objective["output penalty"] = nnet.output_penalty_val
    if Cfg.svdd_loss and not pretrain:
        objective["R"] = nnet.R

    plot_line(objective,
              title="Test objective " + title_suffix,
              xlabel=xlabel,
              ylabel="Objective",
              export_pdf=(xp_path + "/" + file_prefix + "obj_test"))


def plot_accuracy(nnet, xp_path, title_suffix, xlabel, file_prefix):
    """
    plot accuracy of train and validation/test set per epoch.
    """

    # hacky extraction of monitoring values for bedroom dataset
    if nnet.data.dataset_name == "bedroom":
        acc = OrderedDict([("val", nnet.acc_train), ("test", nnet.acc_val)])
    else:
        acc = OrderedDict([("train", nnet.acc_train), ("test", nnet.acc_val)])

    plot_line(acc, title="Accuracy " + title_suffix, xlabel=xlabel,
              ylabel="Accuracy (%) ", y_min=-5, y_max=105,
              export_pdf=(xp_path + "/" + file_prefix + "accuracy"))


def plot_auc(nnet, xp_path, title_suffix, xlabel, file_prefix):
    """
    plot auc time series of train and validation/test set.
    """
    auc = OrderedDict()
    if sum(nnet.data._y_train) > 0:
        auc['train'] = nnet.auc_train
    auc['test'] = nnet.auc_val

    plot_line(auc, title="AUC " + title_suffix, xlabel=xlabel, ylabel="AUC",
              y_min=-0.05, y_max=1.05,
              export_pdf=(xp_path + "/" + file_prefix + "auc"))


def plot_parameter_norms(nnet, xp_path, title_suffix, xlabel, file_prefix):
    """
    plot norms of network parameters (and parameter updates)
    """

    # plot norms of parameters for each unit of dense layers
    params = OrderedDict()

    if Cfg.ocsvm_loss:
        params['W_svm'] = 2 * nnet.Wsvm_norm
        params['rho'] = nnet.rhosvm

    n_layer = 0
    for layer in nnet.trainable_layers:
        if layer.isdense:
            for unit in range(layer.num_units):
                name = "W" + str(n_layer + 1) + str(unit + 1)
                params[name] = nnet.W_norms[n_layer][unit, :]
                if layer.b is not None:
                    name = "b" + str(n_layer + 1) + str(unit + 1)
                    params[name] = nnet.b_norms[n_layer][unit, :]
            n_layer += 1

    plot_line(params, title="Norms of network parameters " + title_suffix,
              xlabel=xlabel, ylabel="Norm", log_scale=True,
              export_pdf=(xp_path + "/" + file_prefix + "param_norms"))

    # plot norms of parameter differences between updates for each layer
    params = OrderedDict()

    n_layer = 0
    for layer in nnet.trainable_layers:
        if layer.isdense | layer.isconv:
            name = "dW" + str(n_layer + 1)
            params[name] = nnet.dW_norms[n_layer]
            if layer.b is not None:
                name = "db" + str(n_layer + 1)
                params[name] = nnet.db_norms[n_layer]
            n_layer += 1
        if layer.issvm and Cfg.ocsvm_loss:
            name = "dWsvm"
            params[name] = nnet.dWsvm_norm
            name = "drho"
            params[name] = nnet.drhosvm_norm

    plot_line(params,
              title="Absolute differences of parameter updates " + title_suffix,
              xlabel=xlabel, ylabel="Norm", log_scale=True,
              export_pdf=(xp_path + "/" + file_prefix + "param_diff_norms"))


def plot_scores(nnet, xp_path, title_suffix, xlabel, file_prefix):
    """
    plot scores of train and validation/test set.
    """

    # plot summary of train scores

    # hacky extraction of monitoring values for bedroom dataset
    if nnet.data.dataset_name == "bedroom":
        scores = OrderedDict([
            ('normal', nnet.scores_train[nnet.data._y_val == 0])])
        if sum(nnet.data._y_val) > 0:
            scores['outlier'] = nnet.scores_train[nnet.data._y_val == 1]
        title = "Summary of validation scores " + title_suffix
    else:
        scores = OrderedDict([
            ('normal', nnet.scores_train[nnet.data._y_train == 0])])
        if sum(nnet.data._y_train) > 0:
            scores['outlier'] = nnet.scores_train[nnet.data._y_train == 1]
        title = "Summary of train scores " + title_suffix

    plot_five_number_summary(scores,
                             title=title,
                             xlabel=xlabel, ylabel="Score",
                             export_pdf=(xp_path + "/"
                                         + file_prefix
                                         + "scores_train"))

    # plot summary of validation/test scores

    # hacky extraction of monitoring values for bedroom dataset
    if nnet.data.dataset_name == "bedroom":
        scores = OrderedDict(
            [('normal', nnet.scores_val[nnet.data._y_test == 0]),
             ('outlier', nnet.scores_val[nnet.data._y_test == 1])])
    else:
        scores = OrderedDict(
            [('normal', nnet.scores_val[nnet.data._y_val == 0]),
             ('outlier', nnet.scores_val[nnet.data._y_val == 1])])

    plot_five_number_summary(scores,
                             title="Summary of test scores " + title_suffix,
                             xlabel=xlabel, ylabel="Score",
                             export_pdf=(xp_path + "/"
                                         + file_prefix
                                         + "scores_test"))


def plot_representation_norms(nnet, xp_path, title_suffix, xlabel, file_prefix):
    """
    plot norms of feature representations of train and validation/test set.
    """

    ylab = "Feature representation norm"

    # plot summary of train feature representation norms

    # hacky extraction of monitoring values for bedroom dataset
    if nnet.data.dataset_name == "bedroom":
        title = "Summary of validation feature rep. norms " + title_suffix
        rep_norm = OrderedDict([
            ('normal', nnet.rep_norm_train[nnet.data._y_val == 0])])
        if sum(nnet.data._y_val) > 0:
            rep_norm['outlier'] = nnet.rep_norm_train[nnet.data._y_val == 1]
    else:
        title = "Summary of train feature rep. norms " + title_suffix
        rep_norm = OrderedDict([
            ('normal', nnet.rep_norm_train[nnet.data._y_train == 0])])
        if sum(nnet.data._y_train) > 0:
            rep_norm['outlier'] = nnet.rep_norm_train[nnet.data._y_train == 1]

    plot_five_number_summary(rep_norm, title=title, xlabel=xlabel, ylabel=ylab,
                             export_pdf=(xp_path + "/"
                                         + file_prefix
                                         + "rep_norm_train"))

    # plot summary of validation/test feature representation norms

    # hacky extraction of monitoring values for bedroom dataset
    if nnet.data.dataset_name == "bedroom":
        rep_norm = OrderedDict([
            ('normal', nnet.rep_norm_val[nnet.data._y_test == 0]),
            ('outlier', nnet.rep_norm_val[nnet.data._y_test == 1])])
    else:
        rep_norm = OrderedDict([
            ('normal', nnet.rep_norm_val[nnet.data._y_val == 0]),
            ('outlier', nnet.rep_norm_val[nnet.data._y_val == 1])])

    title = "Summary of test feature rep. norms " + title_suffix
    plot_five_number_summary(rep_norm, title=title, xlabel=xlabel, ylabel=ylab,
                             export_pdf=(xp_path + "/"
                                         + file_prefix
                                         + "rep_norm_test"))
