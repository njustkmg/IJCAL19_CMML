from __future__ import print_function
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindvision.classification.models import resnet101
from mindspore.ops import operations as P


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.resnet = resnet101(pretrained=True)
        self.fc_retrieval = nn.Dense(1000, 20)

        self.att = nn.Dense(1000, 1)
        self.fc_identify = nn.Dense(1000, 2)
        # self.matmul = ops.matmul
        self.cast = P.Cast()
        self.expend = mindspore.ops.ExpandDims()

    def construct(self, x):

        x = self.resnet(x)
        # print(x)
        att = self.att(x).transpose(1, 0)
        # print(att.shape)
        x = ops.matmul(att, x)
        # print(x.shape)
        identify = self.fc_identify(x)

        # preds = self.cast(identify, mindspore.float32)
        # preds = ops.Argmax(output_type=mindspore.int32)(preds)
        # preds = self.cast(preds, mindspore.float32)
        # print(type(preds))
        # preds = self.expend(preds, 0)
        return identify
