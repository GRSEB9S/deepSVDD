import time
import numpy as np

from config import Configuration as Cfg
from utils.monitoring import performance
from utils.visualization.diagnostics_plot import plot_diagnostics


def train_network(nnet):

    if nnet.data.dataset_name == "bedroom":
        train_network_bedroom(nnet)
        return
    if Cfg.reconstruction_loss:
        nnet.ae_n_epochs = nnet.n_epochs
        train_autoencoder(nnet)
        return

    print("Starting training with %s" % nnet.sgd_solver)
    # save initial network parameters for diagnostics
    nnet.save_initial_parameters()

    # initialize diagnostics for first epoch (detailed diagnostics per batch)
    nnet.initialize_diagnostics(Cfg.n_batches + 1)

    # initialize early compression and c for deep SVDD
    if Cfg.svdd_loss:
        if Cfg.early_compression:
            nnet.R_ec_seq = np.linspace(0.5, (1 - Cfg.nu.get_value()),
                                        Cfg.ec_n_epochs)
        if Cfg.c_mean_init:
            initialize_c_as_mean(nnet, Cfg.c_mean_init_n_batches)

    for epoch in range(nnet.n_epochs):

        # get copy of current network parameters to track differences between
        # epochs
        nnet.copy_parameters()

        if nnet.solver == 'nesterov' and (epoch + 1) % 10 == 0:
            lr = Cfg.floatX(Cfg.learning_rate.get_value() / 10.)
            Cfg.learning_rate.set_value(lr)

        # In each epoch, we do a full pass over the training data:
        start_time = time.time()

        # learning rate decay
        if Cfg.lr_decay:
            decay_learning_rate(nnet, epoch)

        # train on epoch
        i_batch = 0
        for batch in nnet.data.get_epoch_train():

            # Evaluation before training
            if (epoch == 0) and (i_batch == 0):
                _, _ = performance(nnet, which_set='train', epoch=i_batch)
                _, _ = performance(nnet, which_set='val', epoch=i_batch)

            # train
            inputs, targets, _ = batch
            if Cfg.svdd_loss:
                if Cfg.early_compression and (epoch < Cfg.ec_n_epochs):
                    _, _ = nnet.backprop_without_R(inputs, targets)
                elif Cfg.hard_margin:
                    _, _ = nnet.backprop_ball(inputs, targets)
                else:
                    _, _ = nnet.backprop(inputs, targets)
            else:
                _, _ = nnet.backprop(inputs, targets)

            # Get detailed diagnostics (per batch) for the first epoch
            if epoch == 0:
                _, _ = performance(nnet, which_set='train', epoch=i_batch+1)
                _, _ = performance(nnet, which_set='val', epoch=i_batch+1)
                nnet.copy_parameters()
                i_batch += 1

        if epoch == 0:
            # Plot diagnostics for first epoch
            plot_diagnostics(nnet, Cfg.xp_path, Cfg.title_suffix,
                             xlabel="Batches", file_prefix="e1_")
            # Re-initialize diagnostics on epoch level
            nnet.initialize_diagnostics(nnet.n_epochs)
            nnet.copy_initial_parameters_to_cache()

        # Performance on training set (use forward pass with deterministic=True)
        # to get exact training objective
        train_objective, train_accuracy = performance(nnet, which_set='train',
                                                      epoch=epoch, print_=True)

        # Adjust radius R of the SVDD objective (early compression)
        if Cfg.svdd_loss:
            if Cfg.early_compression and (epoch < (Cfg.ec_n_epochs - 1)):
                out_idx = int(np.floor(
                    nnet.data.n_train * (1 - nnet.R_ec_seq[0])))
                sort_idx = nnet.scores_train[:, epoch].argsort()
                R_new = (nnet.scores_train[sort_idx, epoch][-out_idx]
                         + nnet.Rvar.get_value())
                nnet.Rvar.set_value(Cfg.floatX(R_new))
            if Cfg.hard_margin:
                # set R to be the (1-nu)-th quantile of distances
                out_idx = int(np.floor(
                    nnet.data.n_train * Cfg.nu.get_value()))
                sort_idx = nnet.scores_train[:, epoch].argsort()
                R_new = (nnet.scores_train[sort_idx, epoch][-out_idx]
                         + nnet.Rvar.get_value())
                nnet.Rvar.set_value(Cfg.floatX(R_new))

        # Adjust radius R of the SVDD objective (fastR)
        if Cfg.svdd_loss and Cfg.fastR:
            out_idx = int(np.floor(nnet.data.n_train * Cfg.nu.get_value()))
            sort_idx = nnet.scores_train[:, epoch].argsort()
            R_new = (nnet.scores_train[sort_idx, epoch][-out_idx]
                     + nnet.Rvar.get_value())
            if R_new < nnet.Rvar.get_value():
                nnet.Rvar.set_value(Cfg.floatX(R_new))

        # Performance on validation set
        val_objective, val_accuracy = performance(nnet, which_set='val',
                                                  epoch=epoch, print_=True)

        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, nnet.n_epochs, time.time() - start_time))
        print('')

        # log performance
        nnet.log['train_objective'].append(train_objective)
        nnet.log['train_accuracy'].append(train_accuracy)
        nnet.log['val_objective'].append(val_objective)
        nnet.log['val_accuracy'].append(val_accuracy)
        nnet.log['time_stamp'].append(time.time() - nnet.clock)

        # save model as required
        if epoch + 1 == nnet.save_at:
            nnet.dump_weights(nnet.save_to)

    # save train time
    nnet.train_time = time.time() - nnet.clock

    # test
    test_objective, test_accuracy = performance(nnet, which_set='test',
                                                print_=True)
    nnet.stop_clock()
    nnet.test_time = time.time() - (nnet.train_time + nnet.clock)

    # save final weights (and best weights in case of two-class dataset)
    nnet.dump_weights("{}/weights_final.p".format(Cfg.xp_path))
    if nnet.data.n_classes == 2:
        nnet.dump_best_weights("{}/weights_best_ep.p".format(Cfg.xp_path))

    # log final performance
    nnet.log['test_objective'] = test_objective
    nnet.log['test_accuracy'] = test_accuracy


def train_network_bedroom(nnet):

    assert nnet.data.dataset_name == "bedroom"

    if Cfg.c_mean_init:
        initialize_c_as_mean_bedroom(nnet)

    nnet.initialize_diagnostics(Cfg.bedroom_n_train_samples /
                                Cfg.bedroom_monitor_interval)
    nnet.save_initial_parameters()

    print("Starting training with %s" % nnet.sgd_solver)

    i_monitor = 0

    # Iteration only over batches (online learning)
    for batch in range(Cfg.n_batches):

        start_time = time.time()

        # get copy of current network parameters to track differences
        nnet.copy_parameters()

        # load train batch
        nnet.data.load_train_batch_bedroom(batch)

        # learning rate decay
        if Cfg.lr_decay:
            decay_learning_rate(nnet, batch)

        # train
        if Cfg.hard_margin:
            _, _ = nnet.backprop_ball(nnet.data._X_train_batch,
                                      nnet.data._y_train)
        else:
            _, _ = nnet.backprop(nnet.data._X_train_batch, nnet.data._y_train)

        # monitor performance on validation and test set
        if ((batch + 1) * Cfg.batch_size) % Cfg.bedroom_monitor_interval == 0:

            _, _ = performance(nnet, which_set='val',
                               epoch=i_monitor, print_=True)

            if Cfg.hard_margin:
                # set R to be the (1-nu)-th quantile of distances from val set
                out_idx = int(np.floor(nnet.data.n_val * Cfg.nu.get_value()))
                # TODO clean-up hacky extraction for bedroom dataset
                sort_idx = nnet.scores_train[:, i_monitor].argsort()
                R_new = (nnet.scores_train[sort_idx, i_monitor][-out_idx]
                         + nnet.Rvar.get_value())
                nnet.Rvar.set_value(Cfg.floatX(R_new))

            start_test_time = time.time()
            _, _ = performance(nnet, which_set='test',
                               epoch=i_monitor, print_=True)
            nnet.test_time = time.time() - start_test_time

            i_monitor += 1

        print("Batch {} of {} took {:.3f}s".format(
            batch + 1, Cfg.n_batches, time.time() - start_time))
        print('')

    # save train time
    nnet.train_time = time.time() - nnet.clock

    # save final weights (and best weights in case of two-class dataset)
    nnet.dump_weights("{}/weights_final.p".format(Cfg.xp_path))
    if nnet.data.n_classes == 2:
        nnet.dump_best_weights("{}/weights_best_ep.p".format(Cfg.xp_path))


def decay_learning_rate(nnet, epoch):
    """
    decay the learning rate after epoch specified in Cfg.lr_decay_after_epoch
    """

    # only allow decay for non-adaptive solvers
    assert nnet.solver in ("sgd", "momentum", "adam")

    if epoch >= Cfg.lr_decay_after_epoch:
        lr_new = ((Cfg.lr_decay_after_epoch / Cfg.floatX(epoch))
                  * nnet.learning_rate_init)
        Cfg.learning_rate.set_value(Cfg.floatX(lr_new))
    else:
        return


def initialize_c_as_mean(nnet, n_batches, eps=0.1):
    """
    initialize c as the mean of the final layer representations from all samples
    propagated in n_batches
    """

    print("Initializing c...")

    # number of batches (and thereby samples) to initialize from
    if str(n_batches) == "all":
        n_batches = Cfg.n_batches
    if n_batches > Cfg.n_batches:
        n_batches = Cfg.n_batches

    rep_list = list()

    i_batch = 0
    for batch in nnet.data.get_epoch_train():
        inputs, targets, _ = batch

        if i_batch == n_batches:
            break

        _, _, _, _, _, b_rep, _, _, _ = nnet.forward(inputs, targets)
        rep_list.append(b_rep)

        i_batch += 1

    reps = np.concatenate(rep_list, axis=0)
    c = np.mean(reps, axis=0)

    # If c_i is too close to 0 in dimension i, set to +-eps.
    # Reason: a zero unit can be trivially matched with zero weights.
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps

    nnet.cvar.set_value(c)

    # initialize R at the (1-nu)-th quantile of distances
    dist_init = np.sum((reps - c) ** 2, axis=1)
    out_idx = int(np.floor(len(reps) * Cfg.nu.get_value()))
    sort_idx = dist_init.argsort()
    nnet.Rvar.set_value(Cfg.floatX(dist_init[sort_idx][-out_idx]))

    print("c initialized.")


def initialize_c_as_mean_bedroom(nnet):
    """
    initialize c as the mean of the final layer representations from all
    validation set samples
    """

    print("Initializing c...")

    rep_list = list()

    for batch in nnet.data.get_epoch_val():
        inputs, targets, _ = batch
        _, _, _, _, _, b_rep, _, _, _ = nnet.forward(inputs, targets)
        rep_list.append(b_rep)

    reps = np.concatenate(rep_list, axis=0)
    c = np.mean(reps, axis=0)
    nnet.cvar.set_value(c)

    # initialize R at the (1-nu)-th quantile of distances
    dist_init = np.sum((reps - c) ** 2, axis=1)
    out_idx = int(np.floor(len(reps) * Cfg.nu.get_value()))
    sort_idx = dist_init.argsort()
    nnet.Rvar.set_value(Cfg.floatX(dist_init[sort_idx][-out_idx]))

    print("c initialized.")


def train_autoencoder(nnet):

    if Cfg.ae_diagnostics:
        nnet.initialize_ae_diagnostics()

    print("Starting training with %s" % nnet.sgd_solver)

    for epoch in range(nnet.ae_n_epochs):

        start_time = time.time()

        # In each epoch, we do a full pass over the training data:
        l2 = 0
        batches = 0
        train_err = 0
        train_sparse = 0
        train_scores = np.empty(nnet.data.n_train)

        for batch in nnet.data.get_epoch_train():
            inputs, _, batch_idx = batch
            start_idx = batch_idx * Cfg.batch_size
            stop_idx = min(nnet.data.n_train, start_idx + Cfg.batch_size)

            err, l2, b_sparsity, b_scores = nnet.ae_backprop(inputs)

            train_err += err * inputs.shape[0]
            train_sparse += b_sparsity
            train_scores[start_idx:stop_idx] = b_scores.flatten()
            batches += 1

        train_sparse /= batches

        # Performance on validation set
        batches = 0
        val_err = 0
        val_sparse = 0
        val_scores = np.empty(nnet.data.n_val)

        for batch in nnet.data.get_epoch_val():
            inputs, _, batch_idx = batch
            start_idx = batch_idx * Cfg.batch_size
            stop_idx = min(nnet.data.n_val, start_idx + Cfg.batch_size)

            err, l2, b_sparsity, b_scores, _ = nnet.ae_forward(inputs)

            val_err += err * inputs.shape[0]
            val_sparse += b_sparsity
            val_scores[start_idx:stop_idx] = b_scores.flatten()
            batches += 1

        val_sparse /= batches

        # print results for epoch
        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, nnet.ae_n_epochs,
                                                   time.time() - start_time))
        train_err /= nnet.data.n_train
        val_err /= nnet.data.n_val
        print("{:32} {:.5f}".format("Train error:", train_err))
        print("{:32} {:.5f}".format("Test error:", val_err))

        # save diagnostics if specified
        if Cfg.ae_diagnostics:
            nnet.save_ae_diagnostics(epoch, train_err, val_err,
                                     train_sparse, val_sparse,
                                     train_scores, val_scores, l2)

    # save weights
    if Cfg.pretrain:
        nnet.dump_weights("{}/ae_pretrained_weights.p".format(Cfg.xp_path),
                          pretrain=True)
    else:
        nnet.dump_weights("{}/weights_final.p".format(Cfg.xp_path))

    # plot diagnostics if specified
    if Cfg.ae_diagnostics:

        if Cfg.pretrain:
            from utils.visualization.diagnostics_plot import plot_ae_diagnostics
            from utils.visualization.filters_plot import plot_filters

            # common suffix for plot titles
            str_lr = "lr = " + str(nnet.ae_learning_rate)
            C = int(Cfg.C.get_value())
            if not Cfg.weight_decay:
                C = None
            str_C = "C = " + str(C)
            title_suffix = "(" + nnet.ae_solver + ", " + str_C + ", " + str_lr + ")"

            # plot diagnostics
            plot_ae_diagnostics(nnet, Cfg.xp_path, title_suffix)

            # plot filters
            plot_filters(nnet, Cfg.xp_path, title_suffix, file_prefix="ae_",
                         pretrain=True)

        # if image data plot some random reconstructions
        if nnet.data._X_train.ndim == 4:
            from utils.visualization.mosaic_plot import plot_mosaic
            n_img = 32
            random_idx = np.random.choice(nnet.data.n_train, n_img,
                                          replace=False)
            _, _, _, _, reps = nnet.ae_forward(nnet.data._X_train[random_idx,
                                                                  ...])

            title = str(n_img) + " random autoencoder reconstructions"
            plot_mosaic(reps, title=title,
                        export_pdf=(Cfg.xp_path + "/ae_reconstructions"))
