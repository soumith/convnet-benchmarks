import chainer
import chainer.functions as F


class vgga(chainer.FunctionSet):
    insize = 224

    def __init__(self):
        super(vgga, self).__init__(
            conv1=F.Convolution2D(  3,  64, 3, stride=1, pad=1),
            conv2=F.Convolution2D( 64, 128, 3, stride=1, pad=1),
            conv3=F.Convolution2D(128, 256, 3, stride=1, pad=1),
            conv4=F.Convolution2D(256, 256, 3, stride=1, pad=1),
            conv5=F.Convolution2D(256, 512, 3, stride=1, pad=1),
            conv6=F.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv7=F.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv8=F.Convolution2D(512, 512, 3, stride=1, pad=1),
            fc6=F.Linear(512 * 7 * 7, 4096),
            fc7=F.Linear(4096, 4096),
            fc8=F.Linear(4096, 1000),
        )

    def forward(self, x_data, y_data, train=True):
        x = chainer.Variable(x_data, volatile=False)
        t = chainer.Variable(y_data, volatile=False)

        h = F.max_pooling_2d(F.relu(self.conv1(x)), 2, stride=2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 2, stride=2)
        h = F.relu(self.conv3(h))
        h = F.max_pooling_2d(F.relu(self.conv4(h)), 2, stride=2)
        h = F.relu(self.conv5(h))
        h = F.max_pooling_2d(F.relu(self.conv6(h)), 2, stride=2)
        h = F.relu(self.conv7(h))
        h = F.max_pooling_2d(F.relu(self.conv8(h)), 2, stride=2)
        h = F.relu(self.fc6(h))
        h = F.relu(self.fc7(h))
        h = self.fc8(h)
        return F.softmax_cross_entropy(h, t), F.accuracy(h, t)
