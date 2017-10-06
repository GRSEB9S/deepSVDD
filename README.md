# Deep Support Vector Data Description
A deep one-class classification based method for the task of high-dimensional anomaly detection for large-scale applications.

This repository includes the code for the experiments carried out for the Master’s Thesis “Deep Support Vector Data Description” by Lukas Ruff, Humboldt University of Berlin.


# Disclosure
The implementation is based on the repository [https://github.com/oval-group/pl-cnn](https://github.com/oval-group/pl-cnn), which is licensed under the MIT license. The *pl-cnn* repository is an implementation of the paper [Trusting SVM for Piecewise Linear CNNs](https://arxiv.org/abs/1611.02185) by Leonard Berrada, Andrew Zisserman and M. Pawan Kumar, which was an initial inspiration for the topic of this thesis.


# Requirements
This code has been written in `Python 2.7` and requires the packages listed in `requirements.txt` in the denoted versions sourced in a virtual environment.


# Repository organization

## data

Contains the data. The use of the following data sets is implemented:
* MNIST ([http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/))
* CIFAR-10 ([https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html))
* Bedroom ([http://lsun.cs.princeton.edu/2017/](http://lsun.cs.princeton.edu/2017/))

To run the experiments, the data sets have to be downloaded from the original sources in their original formats to the `data` folder.

## src

This directory contains the python code.

## log

This is where experiments and models are logged.


# To reproduce results

Change working directory to `src` and make sure that the standard data sets are downloaded in `data`.
`src` contains two subfolders `experiments` and `scripts`. `experiments` includes the bash scripts to reproduce the experiments reported in the thesis which uses scripts from the `scripts` folder.

To run the MNIST deep SVDD momentum experiments, for example, run 

`sh experiments/mnist_svdd_experiments_momentum.sh`

from the `src` working directory having the requirements loaded in a virtual environment.
If you want to run your own experiments, use the scripts provided in the `scripts` directory.

The CIFAR-10 and Bedroom experiments were run on a GPU by setting the device-argument `gpu1` in the scripts. If you would like to run the experiments on the CPU set the device-argument `cpu`, but expect the experiments to take long time.


# Contact

If there are any problems or questions, feel free to contact!