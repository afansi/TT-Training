from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Sequential


def create_model(
        input_shape=(224, 224, 3),
        conv_list=[(2, 64), (2, 128), (3, 256), (3, 512), (3, 512)],
        dense_list=[4096, 4096, 1000],
        kernel_size=3,
        strides=1,
        pool_size=2,
        dropout_rate=0.1,
        output_activation='sigmoid',
        layer_activation='relu'):
    """ Create a model that will be used for training.
        Args:
            input_shape (tuple of ints, optional): shape of input data.
                Default: (224, 224, 3)
            conv_list (list of pairs of ints):each entry (num, size) represents
                a group of convolutions. num is the number of convolutions
                while size is the number of filters (dimensionality of the
                output space). After each group, a max pooling operation is
                performed. Default: [(2, 64), (2, 128), (3, 256), (3, 512),
                (3, 512)]
            dense_list (list of int): the dimensions of dense layers after
                the convolution ones. Default: [4096, 4096, 1000]
            kernel_size (int): the kernel of the convolution layers. Default: 3
            strides (int): the strides of the convolution layers. Default: 1
            pool_size (int): pool size of the max pooling layers. Default: 2
            dropout_rate (float): dropout rate. Droupout is applied within the
                block of dense layers. Default: 0.1
            output_activation (str): Activation function of the output layer.
                Default: 'sigmoid'
            layer_activation (str): Activation function of the hidden layers.
                Default: 'relu'

        Outputs:
            model (tf.keras.Model): the model to be trained.
    """

    assert len(dense_list) > 0
    assert len(conv_list) > 0

    layer_list = []
    for d in conv_list:
        num_conv, num_filter = d

        if len(layer_list) == 0:
            layer_list.append(
                Conv2D(
                    num_filter, kernel_size, strides,
                    input_shape=input_shape, padding='same',
                    activation=layer_activation
                )
            )
            num_conv = num_conv - 1

        for i in range(num_conv):
            layer_list.append(
                Conv2D(
                    num_filter, kernel_size, strides,
                    padding='same', activation=layer_activation
                )
            )

        layer_list.append(
            MaxPooling2D(
                pool_size=pool_size, strides=2
            )
        )

    layer_list.append(Flatten())

    for i in range(len(dense_list) - 1):
        layer_list.append(Dropout(rate=dropout_rate))
        layer_list.append(
            Dense(
                dense_list[i], activation=layer_activation
            )
        )

    layer_list.append(
        Dense(
            dense_list[-1], activation=output_activation
        )
    )

    model = Sequential(layer_list)
    return model
