from keras.datasets import mnist # Import mnist from the datasets submodule
import numpy as np
from keras import models
from keras import layers
from keras.utils import to_categorical # Import to_categorical

# prompt: get mnist dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# prompt: train_images is a 3D array. transform the train_images to a 2D arrray by combining the 2 lower dimentions
train_images = train_images.reshape((60000, 28 * 28))

# Convert labels to one-hot encoding
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

network = models.Sequential()
network.add(layers.Dense(128, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

network.summary()

network.fit(train_images, train_labels, epochs=10, batch_size=128)

test_images = test_images.reshape((10000, 28 * 28))

test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)