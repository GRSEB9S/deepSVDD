from datasets.__local__ import implemented_datasets
from datasets.mnist import MNIST_DataLoader
from datasets.cifar10 import CIFAR_10_DataLoader
from datasets.bedroom import Bedroom_DataLoader
from datasets.toy import ToySeq_DataLoader
from datasets.normal import Normal_DataLoader
from datasets.adult import Adult_DataLoader


def load_dataset(learner, dataset_name, pretrain=False):

    assert dataset_name in implemented_datasets

    if dataset_name == "mnist":
        data_loader = MNIST_DataLoader

    if dataset_name == "cifar10":
        data_loader = CIFAR_10_DataLoader

    if dataset_name == "bedroom":
        data_loader = Bedroom_DataLoader

    if dataset_name == "toyseq":
        data_loader = ToySeq_DataLoader

    if dataset_name == "normal":
        data_loader = Normal_DataLoader

    if dataset_name == "adult":
        data_loader = Adult_DataLoader

    # load data with data loader
    learner.load_data(data_loader=data_loader, pretrain=pretrain)

    # check all parameters have been attributed
    learner.data.check_all()
