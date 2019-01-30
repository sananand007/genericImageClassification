## Travis-Build
[![Build Status](https://api.travis-ci.org/sananand007/genericImageClassification.png?branch=master)](https://travis-ci.org/sananand007/genericImageClassification)

# GenericImageClassification
Techniques of generic and optimized Image Classification using python and Tensorflow-GPU

## Datasets

| Dataset Name |      Description | Training Set (size) | Test Set (size) |
|-------------:|-----------------:|---------------------|-----------------|
|    DeepSat-6 | Satellite images | 324k                | 81k             |
|              |                  |                     |                 |

## Results

| Dataset Name |                          Description | Traning Accuracy | Test Accuracy | Type of Norm |
|-------------:|-------------------------------------:|------------------|---------------|--------------|
|    DeepSat-6 |             Slope is less for b-norm | 0.9729           | 0.9692        | Batch Norm   |
|              | g-norm converges faster for training | 0.9704           | 0.9657        | Group Norm   |

![alt text](https://github.com/sananand007/genericImageClassification/blob/master/gnVsbn_Accuracy.png)
![alt text](https://github.com/sananand007/genericImageClassification/blob/master/gnVsbn_Loss.png)

## Requirements
 * Linux Ubuntu 18.04
 * Tensorflow 1.12 with GPU enabled
 * CUDA 10 Toolkit with corresponding NVDIA drivers need to be installed
 * I use a 1080Ti Nvidia GPU - currently state of the art, I do not use SLI, just 1 of these

## Implementing Image Classification.

	- Current Implementation uses data from the DeepSat-6 from kaggle to classify images into the 6 classes used
	- All images are used , with a training set of 324k and test set of 81k
	- .h5 files of training and test sets are saved as numpy arrays using paperspace.com as I have limited RAM to handle the large dataset
	- We use Batch Normalization and Group Normalization techniques to classify faster

## TODO

 * Try with bigger and varied datasets, such as imagenet also and compare performance
 * I am integrating the COCO and CIFAR-100 dataset, to train
