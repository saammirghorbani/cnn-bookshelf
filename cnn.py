import tensorflow as tf

model = tf.keras.models.Sequential()


def build_model():
    # 32x32 input neurons
    # Convolution layer #1
    model.add(tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=[5, 5],
        padding='same',
        data_format='channels_last',
        activation=tf.nn.relu))
    # 28x28 neurons remaining

    # Pooling layer #1
    model.add(tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=2))
    # 14x14 neurons remaining

    # Convolution layer #2
    model.add(tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu))
    # 10x10 neurons remaining

    # Pooling layer #2
    model.add(tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=2))
    # 5x5 neurons remaining

    # Dense Layer
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=1))
    model.compile(optimizer=tf.keras.optimizers.SGD(), loss=tf.keras.losses.binary_crossentropy, metrics=['accuracy'])


def train(inputs, labels):
    # batch size equal to number of inputs per row in image, for visual comparison
    model.fit(x=inputs, y=labels, shuffle=True, batch_size=144)


def test(inputs, labels):
    est_loss, test_acc = model.evaluate(inputs, labels, batch_size=144)
    print('Test accuracy:', test_acc)
