# while true;  do top | awk '/python/ {print $8}' ; sleep 1; done


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
	#C3D conv 2
	#model.add(InputLayer(input_shape=(56, 56, 16, 64)))
        #C3D conv 4
	#model.add(InputLayer(input_shape=(28, 28, 8, 256)))
        #VGG conv 2
	#model.add(InputLayer(input_shape=(220, 220, 64)))
        #VGG conv 9
	model.add(InputLayer(input_shape=(27, 27, 512)))
        #Desne Layers
        # NOTHING HERE

	#model.add(Conv2D(filter, kernal))
        #C3D conv 2
	#model.add(Conv3D(128, (3, 3, 3), activation='relu', padding='same'))
        #C3D conv 4
	#model.add(Conv3D(256, (3, 3, 3), activation='relu', padding='same'))
	#C3D dense 1
	#model.add(Dense(4096, input_shape=(8129,), activation='relu'))
	#VGG conv2
	#model.add(Conv2D(64 ,3))
	#VGG conv9
	model.add(Conv2D(512 ,3))
	
	#X = np.random.random_sample((input_size, input_size, channel))
        #C3D conv2
	#X = np.random.random_sample((56,56,16,64))
        #C3D conv4
	#X = np.random.random_sample((28,28,8,256))
        #C3D dense1
	#X = np.random.random_sample((8129))
	#VGG conv2
	X = np.random.random_sample((220, 220, 64))
	#VGG conv9
	X = np.random.random_sample((27, 27, 512))
	
	model.predict(np.array([X]))
	
	start = time.time()
	for _ in range(10):
	    model.predict(np.array([X]))
	print (time.time() - start) / 10


if __name__ == '__main__':
	    main()
