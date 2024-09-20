import network
import layers
import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical



def preprocess(x,y):
    x = x.reshape(x.shape[0], 28*28)
    x = x.astype("float32") / 255
    y = to_categorical(y)
    y = y.reshape(y.shape[1],y.shape[0])
    return x.T,y

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, y_train = preprocess(x_train, y_train)
x_test, y_test = preprocess(x_test, y_test)
model = network.Network()
model.add_layer(layers.Dense(784, 40))
model.add_layer(layers.Relu())
model.add_layer(layers.Dense(40, 10))
model.add_layer(layers.Relu())

model.train(x_train, y_train, batch_size=64, epochs=100, learning_rate=0.06, verbose=True)

#network.save()

for x,y in zip(x_test, y_test):
    output = model.predict(x_test)
    print('pred:', np.argmax(output), '\ttrue:', np.argmax(y))
