import json
import os
import numpy
import mindspore
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as c_vision
import mindspore.dataset.vision.py_transforms as py_vision
from mindspore.dataset.transforms import c_transforms
from mindspore.dataset.transforms import py_transforms
import mindspore.ops as ops
# from mindspore.dataset.vision import Inter
from PIL import Image
# import io

process = py_transforms.Compose([
    # py_vision.Decode(),
    py_vision.Resize([224,224]),
    py_vision.ToTensor(),
    py_vision.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


class Dataset():
    def __init__(self, split=''):
        self.stdnormal = ops.StandardNormal(seed2=0)
        if split == 'train':

            with open("processed_data/img2label_train.json", "r") as f:
                re_data_img = json.load(f)

        elif split == 'val':

            with open("processed_data/img2label_test.json", "r") as f:
                re_data_img = json.load(f)

        elif split == 'test':

            with open("processed_data/img2label_test.json", "r") as f:
                re_data_img = json.load(f)

        else:
            with open("../img2label.json", "r") as f:
                data_img = json.load(f)

        root_2 = '/home/user/Zyx_relate/Projects_dic/Ms_Ysnaker/processed_data/'

        re_imgs = re_data_img['img_id']
        self.re_imgs = []
        for img in re_imgs:
            tmp_path = []
            for i in img:
                path = os.path.join(root_2, i)
                tmp_path.append(path)
            self.re_imgs.append(tmp_path)

        re_labels = list(re_data_img['label'])
        # re_labels = numpy.array(re_labels)
        # re_labels.dtype = "int32"
        # re_labels = mindspore.Tensor(re_labels, mindspore.int32)
        # print(type(re_labels))
        self.re_label = re_labels
        self.transformers = process

    def __getitem__(self, index):

        re_img_path = self.re_imgs[index]
        re_label = self.re_label[index]
        # re_label = mindspore.Tensor(re_label, mindspore.int32)
        # print(re_img_path)
        re_data = mindspore.Tensor(numpy.empty((3, 3, 224, 224)))
        # print(re_img_path[0])
        # print(len(re_img_path))
        # try:
        for i in range(len(re_img_path)):
            re_pil_img = Image.open(re_img_path[i])
            re_data[i] = self.transformers(re_pil_img)
            print(re_data[i])

        # except:
        #     re_data = self.stdnormal((3, 3, 224, 224))

        return re_data, re_label

    def __len__(self):
        return len(self.re_imgs)
