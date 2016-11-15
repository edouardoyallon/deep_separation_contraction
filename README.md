# Building a Regular Classification Boundary with Deep Networks
This is the code for the CVPR17 paper  "Building a Regular Classification Boundary with Deep Networks" by Edouard Oyallon. A large part of the code is inspired from https://github.com/bgshih/tf_resnet_cifar yet it has been modified a lot.

To run all the experiments and obtain the figure of the paper, you can simply do:

bash script_nonlinearity_alpha.bash
python build_figure_paper.py

The best accuracy on CIFAR10 should be 95.4, and on CIFAR100 it should be 79.6, with n_channel equal to 512, alpha=1.0.

# Acknowledgement
Code modified by Edouard Oyallon
