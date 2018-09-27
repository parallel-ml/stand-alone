from keras.layers import Dense, InputLayer
from keras.models import Sequential
import argparse
import numpy as np
import time


parser = argparse.ArgumentParser('Run layer with customized input and kernal.')
parser.add_argument('input', type=int, help='Input size.')
parser.add_argument('output', type=int, help='Output size.')
args = parser.parse_args()

input_shape, output_shape = args.input, args.output

model = Sequential()
model.add(InputLayer(input_shape=(input_shape,)))
model.add(Dense(output_shape))

X = np.random.random_sample((input_shape))

model.predict(np.array([X]))

start = time.time()
for _ in range(10):
    model.predict(np.array([X]))
print (time.time() - start) / 10