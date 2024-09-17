import network
import layers
from keras.datasets import mnist
from keras.utils import to_categorical

train_data = mnist.load_data()[0][0]
flatten_train_data = []
for data in train_data:
    flatten_train_data.append(data.flatten())
train_labels = to_categorical(mnist.load_data()[0][1])

network = network.Network(batch_size=64)

network.add_layer(layers.Dense(784, 10))

network.train(flatten_train_data, train_labels)

