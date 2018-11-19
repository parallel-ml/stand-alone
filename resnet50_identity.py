from keras import layers
from keras.layers import Conv2D, Activation, Input
from keras.models import Model
import numpy as np
import time


def identity(input_tensor, kernel_size, filters, name):
    f1, f2, f3 = filters

    x = Conv2D(f1, (1, 1), activation='relu', name=name + 'a_conv')(input_tensor)
    x = Conv2D(f2, (kernel_size, kernel_size), padding='same', activation='relu', name=name + 'b_conv')(x)
    x = Conv2D(f3, (1, 1), name=name + 'c_conv')(x)

    x = Activation('relu')(x)
    return x


def conv(input_tensor, kernel_size, filters, name, strides=(2, 2)):
    f1, f2, f3 = filters

    x = Conv2D(f1, (1, 1), strides=strides, activation='relu', name=name + 'a_conv')(input_tensor)
    x = Conv2D(f2, (kernel_size, kernel_size), padding='same', activation='relu', name=name + 'b_conv')(x)
    x = Conv2D(f3, (1, 1), name=name + 'c_conv')(x)
    shortcut = Conv2D(f3, (1, 1), strides=strides, name=name + 's_conv')(input_tensor)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def predict(input_shape, filter_size, name):
    img_input = Input(input_shape)
    x = identity(img_input, 3, filter_size, 'identity_')
    model = Model(img_input, x)

    test = np.random.random_sample(input_shape)
    start = time.time()
    for _ in range(50):
        model.predict(np.array([test]))
    return '{:s} {:.3f}s\n'.format('X'.join(str(x) for x in input_shape), (time.time() - start) / 50)


with open('stats', 'w+') as f:
    f.write(predict([55, 55, 64], [64, 64, 256], ''))
    f.write(predict([55, 55, 256], [128, 128, 512], ''))
    f.write(predict([28, 28, 512], [256, 256, 1024], ''))
    f.write(predict([14, 14, 1024], [512, 512, 2048], ''))
