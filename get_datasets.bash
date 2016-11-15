#!/bin/bash


if [ ! -d ./data_cifar10/ ]; then
    mkdir data_cifar10;
fi
if [ ! -f ./data_cifar10/cifar-10-python.tar.gz ]; then
    echo "DLing cifar10...";
    wget -P ./data_cifar10/ http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz;
    tar -zxvf ./data_cifar10/cifar-10-python.tar.gz --directory data_cifar10;
    python create_dataset.py --data_dir ./data_cifar10/cifar-10-batches-py --data_name cifar10;
fi

if [ ! -d ./data_cifar100/ ]; then
    mkdir data_cifar100;
fi
if [ ! -f ./data_cifar100/cifar-100-python.tar.gz ]; then
    echo "DLing cifar100...";
    wget -P ./data_cifar100/ http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz;
    tar -zxvf ./data_cifar100/cifar-100-python.tar.gz --directory data_cifar100;
    python create_dataset.py --data_dir ./data_cifar100/cifar-100-python --data_name cifar100;
fi

