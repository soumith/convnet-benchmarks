import theano.tensor as T
from lasagne.layers import InputLayer, DenseLayer, Conv2DLayer,\
    MaxPool2DLayer

image_sz = 231


def build_model(batch_size=128):
    x = T.tensor4('input')
    layer = InputLayer((batch_size, 3, image_sz, image_sz), input_var=x)

    layer = Conv2DLayer(layer, 96, 11, stride=4, pad='valid')
    layer = MaxPool2DLayer(layer, 2)

    layer = Conv2DLayer(layer, 256, 5, pad='valid')
    layer = MaxPool2DLayer(layer, 2)

    layer = Conv2DLayer(layer, 512, 3, pad='same')
    layer = Conv2DLayer(layer, 1024, 3, pad='same')
    layer = Conv2DLayer(layer, 1024, 3, pad='same')
    layer = MaxPool2DLayer(layer, 2)

    layer = DenseLayer(layer, 3072)
    layer = DenseLayer(layer, 4096)
    layer = DenseLayer(layer, 1000, nonlinearity=None)

    return layer, x
