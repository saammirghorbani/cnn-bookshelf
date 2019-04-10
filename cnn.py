import tensorflow as tf

model = tf.keras.models.Sequential()


def train(img_data, labels):
    # 32x32 input neurons
    inputs = img_data

    # Convolution layer #1
    model.add(tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=[5, 5],
        padding='same',
        data_format='channels_last',
        activation=tf.nn.relu))

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
    model.add(tf.keras.layers.MaxPooling2D(pool_size=[10, 10]))
    # 1x1 neuron remaining

    model.compile(optimizer=tf.keras.optimizers.SGD(), loss=tf.losses.sigmoid_cross_entropy, metrics=['accuracy'])
    model.fit(x=inputs, y=labels)


def test(img_data, labels):
    est_loss, test_acc = model.evaluate(img_data, labels)
    print('Test accuracy:', test_acc)
