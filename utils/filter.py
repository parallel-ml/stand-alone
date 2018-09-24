"""
    This module implements single Conv2D layer filter wise split.
    Compared to channel wise split, instead of splitting the filter
    parameter, it keeps the same filter but splits the input. At the
    end, it adds everything back.
"""
from keras.layers import Conv2D, Lambda, Activation
from keras.layers.merge import Add
import keras.backend as K


def split(X, num):
    depth = K.int_shape(X)[-1]
    d, dl = depth / num, depth - (num - 1) * (depth / num)
    boundary = []
    for i in range(num):
        if i != num - 1:
            boundary.append((i * d, (i + 1) * d))
        else:
            boundary.append((depth - dl, depth))
    return Lambda(lambda x: [x[:, :, :, lb:rb] for lb, rb in boundary])(X)


def merge(tensors, activation):
    X = Add()([x for x in tensors])
    if activation is not None:
        X = Activation(activation)(X)
    return X


def conv(tensors, filters, kernal, strides, padding, name):
    return [Conv2D(filters, kernal, strides=strides, padding=padding, use_bias=i + 1 == len(tensors),
                   name=name + '_conv_' + str(i))(x) for i, x in enumerate(tensors)]