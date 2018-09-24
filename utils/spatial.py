"""
    This module implements single Conv2D layer spatial split.
    It provides an example of 2 division and another generalized
    example. The arithmetic technique is discussed in 2 division
    example.
"""
from keras.layers import Conv2D, Lambda, ZeroPadding2D
from keras.layers.merge import Concatenate
import keras.backend as K
import math


def split_xy(X, kernal, strides, padding, num):
    """ A general function for split tensors with different shapes. """
    # take care of padding here and set padding of conv always to be valid
    if padding == 'same':
        wk, hk = kernal
        ws, hs = strides
        _, W, H, _ = K.int_shape(X)
        ow, oh = W / ws, H / hs
        if W % ws != 0:
            ow += 1
        if H % hs != 0:
            oh += 1
        wp, hp = (ow - 1) * ws + wk - W, (oh - 1) * hs + hk - H
        wp, hp = wp if wp >= 0 else 0, hp if hp >= 0 else 0
        X = ZeroPadding2D(padding=((hp / 2, hp - hp / 2), (wp / 2, wp - wp / 2)))(X)

    wk, hk = kernal
    ws, hs = strides
    _, W, H, _ = K.int_shape(X)

    # output size
    ow, oh = (W - wk) / ws + 1, (H - hk) / hs + 1

    # calculate boundary for general chunk
    wchunk, hchunk = ow / num, oh / num
    rw, rh = (wchunk - 1) * ws + wk, (hchunk - 1) * hs + hk

    # calculate special boundary for last chunk
    wlchunk, hlchunk = ow - (num - 1) * wchunk, oh - (num - 1) * hchunk
    lrw, lrh = (wlchunk - 1) * ws + wk, (hlchunk - 1) * hs + hk

    offset = lambda kernals, strides, i: (kernals - strides) * i if kernals - strides > 0 else 0

    # create a list of tuple with boundary (left, right, up, down)
    boundary = []
    for r in range(num):
        for c in range(num):
            if r == num - 1 and c == num - 1:
                boundary.append((W - lrw, W, H - lrh, H))
            elif r == num - 1:
                boundary.append((rw * c - offset(wk, ws, c), rw * c - offset(wk, ws, c) + rw, H - lrh, H))
            elif c == num - 1:
                boundary.append((W - lrw, W, rh * r - offset(hk, hs, r), rh * r - offset(hk, hs, r) + rh))
            else:
                boundary.append(
                    (
                        rw * c - offset(wk, ws, c),
                        rw * c - offset(wk, ws, c) + rw,
                        rh * r - offset(hk, hs, r),
                        rh * r - offset(hk, hs, r) + rh,
                    )
                )

    return Lambda(
        lambda x:
        [x[:, lb:rb, ub:db, :] for lb, rb, ub, db in boundary]
    )(X)


def merge(tensors):
    """
        The merge function will concatenate all inputs vertically and
        then horizontally.
    """
    size = int(math.sqrt(len(tensors)))
    rows = [Concatenate(axis=1)(tensors[k * size:k * size + size]) for k in range(size)]
    return Concatenate(axis=2)(rows)


def conv(tensors, filters, kernal, strides, padding, activation, name):
    layer = Conv2D(filters, kernal, strides=strides, padding=padding, activation=activation, name=name + '_conv')
    return [layer(x) for x in tensors]