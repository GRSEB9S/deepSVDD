import time
import numpy as np

from config import Configuration as Cfg
from utils.visualization.scatter_plot import plot_2Dscatter


def print_obj_and_acc(objective, accuracy, which_set):

    objective_str = '{} objective:'.format(which_set.title())
    accuracy_str = '{} accuracy:'.format(which_set.title())
    print("{:32} {:.5f}".format(objective_str, objective))
    print("{:32} {:.2f}%".format(accuracy_str, accuracy))


def performance(nnet, which_set, epoch=None, print_=False):

    floatX = Cfg.floatX

    objective = 0
    accuracy = 0
    batches = 0
    emp_loss = 0
    sparsity_penalty = 0
    ball_penalty = 0
    output_penalty = 0
    R = 0

    n = 0
    if which_set == 'train':
        n = nnet.data.n_train
    if which_set == 'val':
        n = nnet.data.n_val
    if which_set == 'test':
        n = nnet.data.n_test
    scores = np.empty(n, dtype=floatX)

    rep = np.empty((n, nnet.all_layers[-1].input_shape[1]), dtype=floatX)
    rep_norm = np.empty(n, dtype=floatX)


    for batch in nnet.data.get_epoch(which_set):
        inputs, targets, batch_idx = batch

        start_idx = batch_idx * Cfg.batch_size
        stop_idx = min(n, start_idx + Cfg.batch_size)

        if Cfg.softmax_loss:
            err, acc, b_scores, l2, b_loss = nnet.forward(inputs, targets)
            scores[start_idx:stop_idx] = b_scores[:, 0]
        elif Cfg.ocsvm_loss:
            err, acc, b_scores, l2, l2_out, b_rep, b_rep_norm, b_loss, b_ball = nnet.forward(inputs, targets)
            scores[start_idx:stop_idx] = b_scores[:, 0]
            rep[start_idx:stop_idx, :] = b_rep
            rep_norm[start_idx:stop_idx] = b_rep_norm
            ball_penalty += b_ball
            output_penalty += l2_out
        elif Cfg.svdd_loss:
            err, acc, b_scores, l2, b_sparsity, _, b_rep_norm, b_loss, R = nnet.forward(inputs, targets)
            scores[start_idx:stop_idx] = b_scores.flatten()
            rep_norm[start_idx:stop_idx] = b_rep_norm
            sparsity_penalty += b_sparsity
        else:
            err, acc, b_scores, l2, b_rep_norm, b_loss = nnet.forward(inputs, targets)
            scores[start_idx:stop_idx] = b_scores.flatten()
            rep_norm[start_idx:stop_idx] = b_rep_norm

        objective += err
        accuracy += acc
        emp_loss += b_loss
        batches += 1

    objective /= batches
    accuracy *= 100. / batches
    emp_loss /= batches
    sparsity_penalty /= batches
    ball_penalty /= batches
    output_penalty /= batches

    if print_:
        print_obj_and_acc(objective, accuracy, which_set)
        if Cfg.ocsvm_loss:
            if not Cfg.rho_fixed:
                rho = np.sum(nnet.ocsvm_layer.b.get_value())
                print("{:32} {:.5f}".format('Rho:', rho))

    # save diagnostics for train and validation set

    # hacky extraction of monitoring values for bedroom dataset
    # nasty hack to utilize train/val infrastructure for val/test sets
    # on bedroom dataset:
    if nnet.data.dataset_name == "bedroom":
        if which_set == 'val':
            which_set = 'train'
        if which_set == 'test':
            which_set = 'val'

    if (which_set == 'train') | (which_set == 'val'):
        nnet.save_objective_and_accuracy(epoch, which_set, objective, accuracy)

        if which_set == 'train':
            nnet.save_train_diagnostics(epoch, scores, rep_norm, rep,
                                        emp_loss, sparsity_penalty,
                                        ball_penalty, output_penalty)

            # Save network parameter diagnostics (only once per epoch)
            nnet.save_network_diagnostics(epoch, floatX(l2), floatX(R))

        if which_set == 'val':
            nnet.save_val_diagnostics(epoch, scores, rep_norm, rep,
                                      emp_loss, sparsity_penalty,
                                      ball_penalty, output_penalty)

            # Track results of epoch with highest AUC on validation set
            if nnet.data.n_classes == 2:
                nnet.track_best_results(epoch)

        # Plot final layer representation
        if Cfg.ocsvm_loss and Cfg.plot_rep:
            xp_path = "../log/" + nnet.data.dataset_name + "/data"

            if which_set == 'train':
                data = {"normal": nnet.rep_train[nnet.data._y_train == 0],
                        "outlier": nnet.rep_train[nnet.data._y_train == 1]}
                title_prefix = "Train "
                file_prefix = "/train_rep"
            if which_set == 'val':
                data = {"normal": nnet.rep_val[nnet.data._y_val == 0],
                        "outlier": nnet.rep_val[nnet.data._y_val == 1]}
                title_prefix = "Test "
                file_prefix = "/test_rep"

            title = (title_prefix + "dataset representation: "
                     + nnet.data.dataset_name + " (seed: "
                     + str(Cfg.seed) + ")")
            plot_2Dscatter(data, title=title,
                           export_pdf=(xp_path + file_prefix + str(epoch + 1)))

    return objective, accuracy


def performance_val_avg(nnet, layer):

    C = Cfg.C.get_value()

    val_objective = 0
    val_accuracy = 0

    for batch in nnet.data.get_epoch_val():
        inputs, targets, _ = batch
        err, acc = layer.hinge_avg(inputs, targets)
        val_objective += err
        val_accuracy += acc

    val_objective /= nnet.data.n_val
    val_accuracy *= 100. / nnet.data.n_val

    val_objective += layer.norm_avg() / (2. * C)
    for other_layer in nnet.trainable_layers:
        if other_layer is not layer:
            val_objective += other_layer.norm() / (2. * C)

    return val_objective, val_accuracy


def hinge_avg(nnet, layer):

    train_objective = 0
    train_accuracy = 0
    for batch in nnet.data.get_epoch_train():
        inputs, targets, _ = batch
        new_err, new_acc = layer.hinge_avg(inputs, targets)

        train_objective += new_err
        train_accuracy += new_acc

    train_objective *= 1. / nnet.data.n_train
    train_accuracy *= 100. / nnet.data.n_train

    return train_objective, train_accuracy


def primal_avg(nnet, layer):

    train_objective, train_accuracy = hinge_avg(nnet, layer)

    train_objective += layer.warm_reg_avg()

    return train_objective, train_accuracy


def performance_train_avg(nnet, layer):

    C = Cfg.C.get_value()

    train_objective, train_accuracy = hinge_avg(nnet, layer)

    train_objective += layer.norm_avg() / (2. * C)

    for llayer in nnet.trainable_layers:
        if llayer is not layer:
            train_objective += llayer.norm() / (2. * C)

    return train_objective, train_accuracy


def checkpoint(nnet, layer):

    C = Cfg.C.get_value()

    hinge_loss, train_accuracy = hinge_avg(nnet, layer)

    primal_objective = hinge_loss + layer.warm_reg_avg()

    train_objective = hinge_loss + layer.norm_avg() / (2. * C)

    for other_layer in nnet.trainable_layers:
        if other_layer is not layer:
            train_objective += other_layer.norm() / (2. * C)

    dual_objective = layer.get_dual()

    val_objective, val_accuracy = performance_val_avg(nnet, layer)

    train_objective = float(train_objective)
    train_accuracy = float(train_accuracy)
    val_objective = float(val_objective)
    val_accuracy = float(val_accuracy)
    primal_objective = float(primal_objective)
    hinge_loss = float(hinge_loss)
    dual_objective = float(dual_objective)

    t = time.time() - nnet.clock

    nnet.log['time_stamp'].append(t)
    nnet.log['layer_tag'].append(layer.name)

    nnet.log['train_objective'].append(train_objective)
    nnet.log['train_accuracy'].append(train_accuracy)

    nnet.log['val_objective'].append(val_objective)
    nnet.log['val_accuracy'].append(val_accuracy)

    nnet.log['primal_objective'].append(primal_objective)
    nnet.log['hinge_loss'].append(hinge_loss)
    nnet.log['dual_objective'].append(dual_objective)

    print("Pass %i - Epoch %i - Layer %s" %
          (nnet.pass_, len(nnet.log['time_stamp']), layer.name))
    print_obj_and_acc(train_objective, train_accuracy, which_set='train')
    print_obj_and_acc(val_objective, val_accuracy, which_set='val')
