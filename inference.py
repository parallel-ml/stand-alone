from keras.layers import Conv2D, InputLayer
from keras.models import Sequential
import argparse
import numpy as np
import time


parser = argparse.ArgumentParser('Run layer with customized input and kernal.')
parser.add_argument('channel', type=int, help='Channel size.')
parser.add_argument('kernal', type=int, help='Kernal size.')
parser.add_argument('filter', type=int, help='Filter size.')
args = parser.parse_args()

channel, kernal, filter = args.channel, args.kernal, args.filter

model = Sequential()
model.add(InputLayer(input_shape=(128, 128, channel)))
model.add(Conv2D(filter, kernal))

X = np.random.random_sample((128, 128, channel))

model.predict(np.array([X]))

start = time.time()
for _ in range(10):
    model.predict(np.array([X]))
print (time.time() - start) / 10