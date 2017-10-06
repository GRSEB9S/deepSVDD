import numpy as np
import theano


class Configuration(object):

    floatX = np.float32
    seed = 0

    # Final Layer
    compile_lwsvm = False
    softmax_loss = False
    ocsvm_loss = False
    svdd_loss = False
    svm_loss = False
    reconstruction_loss = False

    # Optimization
    batch_size = 200
    learning_rate = theano.shared(floatX(1e-4), name="learning rate")
    lr_decay = False
    lr_decay_after_epoch = 10
    momentum = theano.shared(floatX(0.9), name="momentum")
    rho = theano.shared(floatX(0.9), name="rho")
    fastR = False

    eps = floatX(1e-8)

    # Network architecture
    leaky_relu = False
    softplus = False
    dropout = False
    dropout_architecture = False

    # Pre-training and autoencoder configuration
    pretrain = False
    ae_loss = "l2"
    ae_weight_decay = True
    ae_C = theano.shared(floatX(1e3), name="ae_C")
    ae_sparsity_penalty = False
    ae_B = theano.shared(floatX(1e3), name="ae_B")
    ae_sparsity_mode = "mean"
    ae_sparsity = theano.shared(floatX(0.05), name="ae_sparsity")

    # Regularization
    C = theano.shared(floatX(1e3), name="C")
    D = theano.shared(floatX(1e2), name="D")
    Wsvm_penalty = False
    weight_decay = True
    include_bias = False
    pow = 2  # power of the p-norm used in weight decay regularizer
    output_penalty = False
    spec_penalty = False
    prod_penalty = False
    bias_offset = False
    sparsity_penalty = False
    sparsity_mode = "mean"
    B = theano.shared(floatX(1e3), name="B")
    sparsity = theano.shared(floatX(0.05), name="sparsity")

    # OC-SVM
    rho_fixed = False
    nu = theano.shared(floatX(.2), name="nu")
    out_frac = floatX(.1)
    ad_experiment = False
    A = theano.shared(floatX(1), name="A")
    ball_penalty = False
    normalize = True

    # SVDD
    c_mean_init = False
    c_mean_init_n_batches = "all"
    hard_margin = False
    gaussian_blob = False
    early_compression = False
    ec_n_epochs = 10

    # Toy data set parameters
    toy_n_train = 1000
    toy_ndim = 2
    toy_net_depth = 2
    toy_net_width = 2
    toy_motif_len = 4
    toy_motif_off = 0
    plot_rep = False

    # Data preprocessing
    weight_dict_init = False
    pca = False
    unit_norm = False
    unit_norm_used = "l2"  # "l2" or "l1"
    z_normalization = False
    gcn = False
    zca_whitening = False

    # MNIST dataset parameters
    mnist_bias = True
    mnist_rep_dim = 16
    mnist_normal = 0
    mnist_outlier = -1

    # CIFAR-10 dataset parameters
    cifar10_bias = True
    cifar10_rep_dim = 32
    cifar10_normal = 1
    cifar10_outlier = -1

    # LSUN bedroom dataset parameters
    bedroom_n_train_samples = int(1e6)
    bedroom_monitor_interval = int(5000)
    bedroom_downscale_pxl = int(64)

    # Plot parameters
    xp_path = "../log/"
    title_suffix = ""

    # SVM parameters
    svm_C = floatX(1.0)
    svm_nu = floatX(0.2)

    # Diagnostics (should diagnostics be retrieved? Training is faster without)
    ae_diagnostics = True
