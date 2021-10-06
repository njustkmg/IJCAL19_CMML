# -*- coding: utf-8 -*-
import paddle
from paddle import nn


class TextfeatureNet(paddle.nn.Layer):
    def __init__(self, neure_num):
        super(TextfeatureNet, self).__init__()
        self.mlp = make_layers(neure_num[:-1])
        self.feature = paddle.nn.Linear(neure_num[-2], neure_num[-1])
        # self.relu = paddle.nn.ReLU()

    def forward(self, x):
        temp_x = self.mlp(x)
        x = self.feature(temp_x)
        # x = self.relu(x)
        return x


class PredictNet(paddle.nn.Layer):
    def __init__(self, neure_num):
        super(PredictNet, self).__init__()
        self.mlp = make_predict_layers(neure_num)
        self.sigmoid = paddle.nn.Sigmoid()

    def forward(self, x):
        y = self.mlp(x)
        y = self.sigmoid(y)
        return y


class AttentionNet(paddle.nn.Layer):
    def __init__(self, neure_num):
        super(AttentionNet, self).__init__()
        self.mlp = make_layers(neure_num[:-1])
        self.attention = paddle.nn.Linear(neure_num[-2], neure_num[-1])

    def forward(self, x):
        temp_x = self.mlp(x)
        y = self.attention(temp_x)
        return y

class ImgNet(paddle.nn.Layer):
    def __init__(self):
        super(ImgNet, self).__init__()
        self.feature = paddle.vision.models.resnet18(pretrained=True)
        self.feature = paddle.nn.Sequential(*list(self.feature.children())[:-1])
        self.fc1 = paddle.nn.Sequential(
            paddle.nn.Linear(512, 128),
        )

    def forward(self, x):
        N = x.shape[0]
        x = self.feature(paddle.reshape(x, shape=[N, 3, 256, 256]))
        x = paddle.reshape(x, shape=[N, 512])
        x = self.fc1(x)
        return x

def make_layers(cfg):
    layers = []
    n = len(cfg)
    input_dim = cfg[0]
    for i in range(1, n):
        output_dim = cfg[i]
        layers += [paddle.nn.Linear(input_dim, output_dim), paddle.nn.ReLU()]
        input_dim = output_dim
    return paddle.nn.Sequential(*layers)

def make_predict_layers(cfg):
    layers = []
    n = len(cfg)
    input_dim = cfg[0]
    for i in range(1, n):
        output_dim = cfg[i]
        layers += [paddle.nn.Linear(input_dim, output_dim)]
        input_dim = output_dim
    return paddle.nn.Sequential(*layers)

def generate_model(Textfeatureparam, Imgpredictparam, Textpredictparam, Attentionparam, Predictparam):
    Textfeaturemodel = TextfeatureNet(Textfeatureparam)
    Imgpredictmodel = PredictNet(Imgpredictparam)
    Textpredictmodel = PredictNet(Textpredictparam)
    Predictmodel = PredictNet(Predictparam)
    Imgmodel = ImgNet()
    Attentionmodel = AttentionNet(Attentionparam)

    return Textfeaturemodel, Imgpredictmodel, Textpredictmodel, Imgmodel, Attentionmodel, Predictmodel
