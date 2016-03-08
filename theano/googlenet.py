import theano.tensor as T
from lasagne.layers import InputLayer, DenseLayer, Conv2DLayer,\
    MaxPool2DLayer, Pool2DLayer, ConcatLayer


image_sz = 224


def _inception(inp, o1s, o2s1, o2s2, o3s1, o3s2, o4s):
    conv1 = Conv2DLayer(inp, o1s, 1)

    conv3_ = Conv2DLayer(inp, o2s1, 1)
    conv3 = Conv2DLayer(conv3_, o2s2, 3, pad='same')

    conv5_ = Conv2DLayer(inp, o3s1, 1)
    conv5 = Conv2DLayer(conv5_, o3s2, 5, pad='same')

    pool_ = MaxPool2DLayer(inp, 3, stride=1, pad=1)
    pool = Conv2DLayer(pool_, o4s, 1)

    return ConcatLayer([conv1, conv3, conv5, pool])


def build_model(batch_size=128):
    x = T.tensor4('input')
    layer = InputLayer((batch_size, 3, image_sz, image_sz), input_var=x)

    conv1 = Conv2DLayer(layer, 64, 7, stride=2, pad='same')
    pool1 = MaxPool2DLayer(conv1, 3, stride=2, pad=1)

    conv2 = Conv2DLayer(pool1, 64, 1, pad='same')
    conv3 = Conv2DLayer(conv2, 192, 3, pad='same')
    pool3 = MaxPool2DLayer(conv3, 3, stride=2, pad=1)

    incept3a = _inception(pool3,    64, 96, 128, 16, 32, 32)
    incept3b = _inception(incept3a, 128, 128, 192, 32, 96, 64)
    pool4 = MaxPool2DLayer(incept3b, 3, stride=2, pad=1)
    incept4a = _inception(pool4,    192,  96, 208, 16, 48, 64)
    incept4b = _inception(incept4a, 160, 112, 224, 24, 64, 64)
    incept4c = _inception(incept4b, 128, 128, 256, 24, 64, 64)
    incept4d = _inception(incept4c, 112, 144, 288, 32, 64, 64)
    incept4e = _inception(incept4d, 256, 160, 320, 32, 128, 128)
    pool5 = MaxPool2DLayer(incept4e, 3, stride=2, pad=1)

    incept5a = _inception(pool5,    256, 160, 320, 32, 128, 128)
    incept5b = _inception(incept5a, 384, 192, 384, 48, 128, 128)
    pool6 = Pool2DLayer(incept5b, 7, stride=1, mode='average_exc_pad')

    layer = DenseLayer(pool6, 1024)
    layer = DenseLayer(layer, 1000, nonlinearity=None)

    return layer, x

