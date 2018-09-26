from keras.layers import Conv3D, Conv2D, InputLayer, Dense
from keras.models import Sequential
import argparse
import numpy as np
import time

from memory_profiler import profile

parser = argparse.ArgumentParser('Run layer with customized input and kernal.')
parser.add_argument('input_size', type=int, help='Filter size.')
parser.add_argument('channel', type=int, help='Channel size.')
parser.add_argument('kernal', type=int, help='Kernal size.')
parser.add_argument('filter', type=int, help='Filter size.')
args = parser.parse_args()

input_size, channel, kernal, filter = args.input_size, args.channel, args.kernal, args.filter

@profile
def main():
	model = Sequential()
	#model.add(InputLayer(input_shape=(input_size, input_size, channel)))
	#model.add(InputLayer(input_shape=(56, 56, 16, 64)))
	#model.add(InputLayer(input_shape=(8192)))

	#model.add(Conv2D(filter, kernal))
	#model.add(Conv3D(128, (3, 3, 3), activation='relu', padding='same'))
	model.add(Dense(4096, input_shape=(4096,), activation='relu'))
	
	#X = np.random.random_sample((input_size, input_size, channel))
	#X = np.random.random_sample((56,56,16,64))
	X = np.random.random_sample((4096))
	
	model.predict(np.array([X]))
	
	start = time.time()
	for _ in range(10):
	    model.predict(np.array([X]))
	print (time.time() - start) / 10


if __name__ == '__main__':
	    main()
