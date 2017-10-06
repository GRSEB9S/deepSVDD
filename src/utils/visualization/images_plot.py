from utils.visualization.mosaic_plot import plot_mosaic


def plot_outliers(model, n, xp_path):
    """
    Plot the first n images of train and test set
    ordered by descending anomaly scores
    """

    # reload images with original scale
    model.data.load_data(original_scale=True)

    idx_sort_train = model.scores_train[:, -1].argsort()[::-1]
    outliers_train = model.data._X_train[idx_sort_train, ...][:n, ...]
    str_samples = "(" + str(n) + " of " + str(int(model.data.n_train)) + ")"
    title = "Train set outliers with ascending scores " + str_samples
    plot_mosaic(outliers_train[::-1], title=title,
                export_pdf=(xp_path + "/outliers_train"))

    idx_sort_test = model.scores_val[:, -1].argsort()[::-1]
    outliers_test = model.data._X_val[idx_sort_test, ...][:n, ...]
    str_samples = "(" + str(n) + " of " + str(int(model.data.n_val)) + ")"
    title = "Test set outliers with ascending scores " + str_samples
    plot_mosaic(outliers_test[::-1], title=title,
                export_pdf=(xp_path + "/outliers_test"))

    # plot with scores at best epoch
    if model.best_weight_dict is not None:

        epoch = model.auc_best_epoch
        str_epoch = "at epoch " + str(epoch) + " "

        idx_sort_train = model.scores_train[:, epoch].argsort()[::-1]
        outliers_train = model.data._X_train[idx_sort_train, ...][:n, ...]
        str_samples = "(" + str(n) + " of " + str(int(model.data.n_train)) + ")"
        title = ("Train set outliers with ascending scores "
                 + str_epoch + str_samples)
        plot_mosaic(outliers_train[::-1], title=title,
                    export_pdf=(xp_path + "/outliers_train_best_ep"))

        idx_sort_test = model.scores_val[:, epoch].argsort()[::-1]
        outliers_test = model.data._X_val[idx_sort_test, ...][:n, ...]
        str_samples = "(" + str(n) + " of " + str(int(model.data.n_val)) + ")"
        title = ("Test set outliers with ascending scores "
                 + str_epoch + str_samples)
        plot_mosaic(outliers_test[::-1], title=title,
                    export_pdf=(xp_path + "/outliers_test_best_ep"))


def plot_normals(model, n, xp_path):
    """
    Plot the first n images of train and test set
    ordered by ascending anomaly scores
    """

    # reload images with original scale
    model.data.load_data(original_scale=True)

    idx_sort_train = model.scores_train[:, -1].argsort()
    normals_train = model.data._X_train[idx_sort_train, ...][:n, ...]
    str_samples = "(" + str(n) + " of " + str(int(model.data.n_train)) + ")"
    title = "Train set examples with ascending scores " + str_samples
    plot_mosaic(normals_train, title=title,
                export_pdf=(xp_path + "/normals_train"))

    idx_sort_test = model.scores_val[:, -1].argsort()
    normals_test = model.data._X_val[idx_sort_test, ...][:n, ...]
    str_samples = "(" + str(n) + " of " + str(int(model.data.n_val)) + ")"
    title = "Test set examples with ascending scores " + str_samples
    plot_mosaic(normals_test, title=title,
                export_pdf=(xp_path + "/normals_test"))

    # plot with scores at best epoch
    if model.best_weight_dict is not None:

        epoch = model.auc_best_epoch
        str_epoch = "at epoch " + str(epoch) + " "

        idx_sort_train = model.scores_train[:, epoch].argsort()
        normals_train = model.data._X_train[idx_sort_train, ...][:n, ...]
        str_samples = "(" + str(n) + " of " + str(int(model.data.n_train)) + ")"
        title = ("Train set examples with ascending scores "
                 + str_epoch + str_samples)
        plot_mosaic(normals_train, title=title,
                    export_pdf=(xp_path + "/normals_train_best_ep"))

        idx_sort_test = model.scores_val[:, epoch].argsort()
        normals_test = model.data._X_val[idx_sort_test, ...][:n, ...]
        str_samples = "(" + str(n) + " of " + str(int(model.data.n_val)) + ")"
        title = ("Test set examples with ascending scores "
                 + str_epoch + str_samples)
        plot_mosaic(normals_test, title=title,
                    export_pdf=(xp_path + "/normals_test_best_ep"))


def plot_outliers_bedroom(model, n, xp_path):
    """
    Plot the first n images of train and test set
    ordered by descending anomaly scores
    """

    # reload images with original scale
    model.data.load_data(original_scale=True)

    idx_sort_train = model.scores_train[:, -1].argsort()[::-1]
    outliers_val = model.data._X_val[idx_sort_train, ...][:n, ...]
    str_samples = "(" + str(n) + " of " + str(int(model.data.n_val)) + ")"
    title = "Validation set outliers with ascending scores " + str_samples
    plot_mosaic(outliers_val[::-1], title=title,
                export_pdf=(xp_path + "/outliers_val"))

    idx_sort_test = model.scores_val[:, -1].argsort()[::-1]
    outliers_test = model.data._X_test[idx_sort_test, ...][:n, ...]
    str_samples = "(" + str(n) + " of " + str(int(model.data.n_test)) + ")"
    title = "Test set outliers with ascending scores " + str_samples
    plot_mosaic(outliers_test[::-1], title=title,
                export_pdf=(xp_path + "/outliers_test"))

    # plot with scores at best epoch
    if model.best_weight_dict is not None:

        epoch = model.auc_best_epoch
        str_epoch = "at epoch " + str(epoch) + " "

        idx_sort_train = model.scores_train[:, epoch].argsort()[::-1]
        outliers_val = model.data._X_val[idx_sort_train, ...][:n, ...]
        str_samples = "(" + str(n) + " of " + str(int(model.data.n_val)) + ")"
        title = ("Validation set outliers with ascending scores "
                 + str_epoch + str_samples)
        plot_mosaic(outliers_val[::-1], title=title,
                    export_pdf=(xp_path + "/outliers_val_best_ep"))

        idx_sort_test = model.scores_val[:, epoch].argsort()[::-1]
        outliers_test = model.data._X_test[idx_sort_test, ...][:n, ...]
        str_samples = "(" + str(n) + " of " + str(int(model.data.n_test)) + ")"
        title = ("Test set outliers with ascending scores "
                 + str_epoch + str_samples)
        plot_mosaic(outliers_test[::-1], title=title,
                    export_pdf=(xp_path + "/outliers_test_best_ep"))


def plot_normals_bedroom(model, n, xp_path):
    """
    Plot the first n images of train and test set
    ordered by ascending anomaly scores
    """

    # reload images with original scale
    model.data.load_data(original_scale=True)

    idx_sort_train = model.scores_train[:, -1].argsort()
    normals_val = model.data._X_val[idx_sort_train, ...][:n, ...]
    str_samples = "(" + str(n) + " of " + str(int(model.data.n_val)) + ")"
    title = "Validation set examples with ascending scores " + str_samples
    plot_mosaic(normals_val, title=title,
                export_pdf=(xp_path + "/normals_val"))

    idx_sort_test = model.scores_val[:, -1].argsort()
    normals_test = model.data._X_test[idx_sort_test, ...][:n, ...]
    str_samples = "(" + str(n) + " of " + str(int(model.data.n_test)) + ")"
    title = "Test set examples with ascending scores " + str_samples
    plot_mosaic(normals_test, title=title,
                export_pdf=(xp_path + "/normals_test"))

    # plot with scores at best epoch
    if model.best_weight_dict is not None:

        epoch = model.auc_best_epoch
        str_epoch = "at epoch " + str(epoch) + " "

        idx_sort_train = model.scores_train[:, epoch].argsort()
        normals_val = model.data._X_val[idx_sort_train, ...][:n, ...]
        str_samples = "(" + str(n) + " of " + str(int(model.data.n_val)) + ")"
        title = ("Validation set examples with ascending scores "
                 + str_epoch + str_samples)
        plot_mosaic(normals_val, title=title,
                    export_pdf=(xp_path + "/normals_val_best_ep"))

        idx_sort_test = model.scores_val[:, epoch].argsort()
        normals_test = model.data._X_test[idx_sort_test, ...][:n, ...]
        str_samples = "(" + str(n) + " of " + str(int(model.data.n_test)) + ")"
        title = ("Test set examples with ascending scores "
                 + str_epoch + str_samples)
        plot_mosaic(normals_test, title=title,
                    export_pdf=(xp_path + "/normals_test_best_ep"))
