import network
import layers
from keras.datasets import mnist
from keras.utils import to_categorical

train_data = mnist.load_data()[0][0]
flatten_train_data = []
for data in train_data:
    flatten_train_data.append(data.flatten())
train_labels = to_categorical(mnist.load_data()[0][1])

test_data = mnist.load_data()[1][0]
flatten_test_data = []
for data in test_data:
    flatten_test_data.append(data.flatten())
test_labels = to_categorical(mnist.load_data()[1][1])

network = network.Network(learning_rate=0.05,batch_size=256)

network.add_layer(layers.Dense(784, 10))

network.train(flatten_train_data, train_labels)

network.save()
"""
model = network.Network.load_model()

pred = model.predict(flatten_test_data)
for i,p in enumerate(pred):
    pred[i] = p.index(max(p))
for i,l in enumerate(test_labels):
    max_val = 0
    max_index = 0
    for j in range(len(l)):
        if l[j] > max_val:
            max_val = l[j]
            max_index = j
    test_labels[i] = max_index
for p, label in zip(pred, test_labels):
    print(p, label)

"""