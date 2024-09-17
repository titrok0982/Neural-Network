#load mnist model
import tensorflow as tf

def train():
    model = tf.keras.models.load_model('models/mnist.h5')
    train_data = tf.keras.datasets.mnist.load_data()[0][0]
    train_labels = tf.keras.utils.to_categorical(tf.keras.datasets.mnist.load_data()[0][1])
    model.fit(train_data, train_labels, epochs=10, validation_split=0.2, batch_size=32)
    test_data = tf.keras.datasets.mnist.load_data()[1][0]
    test_labels = tf.keras.utils.to_categorical(tf.keras.datasets.mnist.load_data()[1][1])
    model.evaluate(test_data, test_labels)
    model.save('models/baseline.h5')
train()