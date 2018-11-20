from keras.layers import Conv2D, Input
from keras.models import Model
import numpy as np
import time


def block(input_tensor, filter):
    f1, f2 = filter
    x = Conv2D(f1, 1, strides=(1, 1), padding='same')(input_tensor)
    x = Conv2D(f2, 3, strides=(1, 1), padding='same')(x)
    return x


def predict(input_shape, filter, name):
    img_input = Input(input_shape)
    x = block(img_input, filter)
    model = Model(img_input, x)

    test = np.random.random_sample(input_shape)
    start = time.time()
    for _ in range(50):
        model.predict(np.array([test]))
    return '{:s} {:.3f}\n'.format(name, (time.time() - start) / 50)


with open('stats', 'w+') as f:
    f.write(predict([109, 109, 64], [128, 128], 'A'))
    f.write(predict([109, 109, 64], [256, 256], 'B'))
    f.write(predict([109, 109, 64], [728, 728], 'C'))
    f.write(predict([109, 109, 64], [728, 728], 'D'))
    f.write(predict([109, 109, 64], [728, 1024], 'E'))
    f.write(predict([109, 109, 64], [1536, 2048], 'F'))
