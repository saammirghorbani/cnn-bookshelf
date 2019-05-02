import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report

model = tf.keras.models.Sequential()


def build_model():
    # 32x32 input neurons
    # Convolution layer #1
    model.add(tf.keras.layers.Conv2D(
        filters=16,
        kernel_size=[5, 5],
        data_format='channels_last',
        activation=tf.nn.relu))
    # 28x28 neurons remaining

    # Pooling layer #1
    model.add(tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=2))
    # 14x14 neurons remaining

    # Convolution layer #2
    model.add(tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=[3, 3],
        data_format='channels_last',
        activation=tf.nn.relu))
    # 12x12 neurons remaining

    # Pooling layer #2
    model.add(tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=2))
    # 6x6 neurons remaining

    # Convolution layer #3
    model.add(tf.keras.layers.Conv2D(
        filters=16,
        kernel_size=[5, 5],
        data_format='channels_last',
        activation=tf.nn.relu))
    # 2x2 neurons remaining

    # Dense Layer
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=4, activation=tf.nn.relu))
    # Reduce chance of overfitting with Dropout layer
    model.add(tf.keras.layers.Dropout(rate=0.3))
    model.add(tf.keras.layers.Dense(units=1, activation=tf.nn.sigmoid))

    model.compile(optimizer=tf.keras.optimizers.SGD(), loss=tf.keras.losses.binary_crossentropy, metrics=['accuracy'])


def train(inputs, labels):
    model.fit(x=inputs, y=labels, shuffle=True, epochs=5)


def test(inputs, labels):
    predicted_classes = model.predict_classes(inputs)
    accuracy = accuracy_score(labels, predicted_classes)
    print("Accuracy: " + str(accuracy))
    """Precision: proportion of predictions that were correct in regards to their labels.
    Recall: proportion of labels that were correctly predicted."""
    print(classification_report(labels, predicted_classes, np.unique(labels)))
    return predicted_classes
