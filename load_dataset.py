import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
import numpy as np


def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', 'r')
    train_set_x_orig = np.array(train_dataset['train_set_x'][:])
    train_set_y_orig = np.array(train_dataset['train_set_y'][:])
    # print(train_set_y_orig)

    # print(train_set_x_orig.shape)
    test_dataset = h5py.File('datasets/test_catvnoncat.h5', 'r')
    test_set_x_orig = np.array(test_dataset['test_set_x'][:])
    test_set_y_orig = np.array(test_dataset['test_set_y'][:])

    classes = np.array(test_dataset['list_classes'][:])
    # print(test_set_x_orig[0])

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def flatten_data():
    train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_dataset()
    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
    # print(train_set_x_flatten.shape)
    # print(test_set_x_flatten.shape)
    # print(train_set_y_orig.shape)
    # print(test_set_y_orig.shape)

    return train_set_x_flatten, train_set_y_orig, test_set_x_flatten, test_set_y_orig, classes


def standardize_data():
    train_set_x_flatten, train_set_y_orig, test_set_x_flatten, test_set_y_orig, classes = flatten_data()

    train_set_x = train_set_x_flatten / 255
    train_set_y = train_set_y_orig
    test_set_x = test_set_x_flatten / 255
    test_set_y = test_set_y_orig

    return train_set_x, train_set_y, test_set_x, test_set_y, classes


