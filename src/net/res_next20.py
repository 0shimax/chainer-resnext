import math
import chainer
import chainer.functions as F
import chainer.links as L


class Cardinality(chainer.Chain):
    def __init__(self, in_size, ch, out_size, stride):
        w = math.sqrt(2)
        super().__init__(
            conv1=L.Convolution2D(in_size, ch, 1, stride, 0, w, nobias=True),
            conv2=L.Convolution2D(ch, ch, 3, 1, 1, w, nobias=True),
            conv3=L.Convolution2D(ch, out_size, 1, 1, 0, w, nobias=True),
        )
    def __call__(self, x):
        h = F.elu(self.conv1(x))
        h = F.elu(self.conv2(h))
        return self.conv3(h)


class Block(chainer.Chain):
    def __init__(self, in_size, ch, out_size, stride, card_size=32):
        super().__init__()
        links = [('c{}'.format(i+1), Cardinality(in_size, ch, out_size, stride)) for i in range(card_size)]
        links += [('x_bypass', L.Convolution2D(in_size, out_size, 1, stride, 0, nobias=True))]
        for l in links:
            self.add_link(*l)
        self.forward = links

    def __call__(self, x):
        h = None
        for name, _ in self.forward:
            f = getattr(self, name)
            h_t = f(x)
            if h is None:
                h = h_t
            else:
                h += h_t
        return F.elu(h)


class LaminationBlock(chainer.Chain):
    def __init__(self, layer, in_size, ch, out_size, stride=1):
        super().__init__()
        links = [('lb0', Block(in_size, ch, out_size, stride))]
        links += [('lb{}'.format(i+1), Block(out_size, ch, out_size, 1)) for i in range(1, layer)]
        for l in links:
            self.add_link(*l)
        self.forward = links

    def __call__(self, x, train):
        for name, _ in self.forward:
            f = getattr(self, name)
            x = f(x)
        return x


class ResNext20(chainer.Chain):
    def __init__(self, n_class, in_ch):
        w = math.sqrt(2)
        super().__init__(
            conv1=L.Convolution2D(in_ch, 128, 7, 2, 3, w, nobias=True),
            res2=LaminationBlock(3, 128, 4, 256, stride=2),
            res3=LaminationBlock(4, 256, 4, 512, stride=2),
            # res4=LaminationBlock(6, 512, 256, 1024),
            # res5=LaminationBlock(3, 1024, 512, 2048),
            conv2=L.Convolution2D(512*21, 4096, 1, pad=0),  # *21
            conv3=L.Convolution2D(4096, n_class, 1, pad=0)
        )
        self.train = True
        self.n_class = n_class
        self.active_learn = False

    def clear(self):
        self.loss = None
        self.accuracy = None

    def __call__(self, x, t):
        x.volatile = not self.train

        h = F.elu(self.conv1(x))
        h = F.max_pooling_2d(h, 3, stride=2)
        h = self.res2(h, self.train)
        h = self.res3(h, self.train)
        # h = self.res4(h, self.train)
        # h = self.res5(h, self.train)
        h = F.spatial_pyramid_pooling_2d(h, 3, F.MaxPooling2D)

        h = F.elu(self.conv2(h))
        h = F.dropout(h, ratio=0.5, train=self.train)
        h = self.conv3(h)
        h = F.reshape(h, (-1, self.n_class))

        self.prob = F.softmax(h)
        if self.active_learn:
            t = mask_gt_for_active_learning(self.prob, t, self.xp, self.n_class)

        self.loss = F.softmax_cross_entropy(h, t)
        self.accuracy = F.accuracy(h, t)
        chainer.report({'loss': self.loss, 'accuracy': self.accuracy}, self)
        return self.loss
