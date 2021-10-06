# -*- coding: utf-8 -*-

import paddle
import pickle
import numpy as np
from PIL import Image

from paddle.io import Dataset
import paddle.vision.transforms as T

class MyDataset(Dataset):
    def __init__(self, imgfilenamerecord, imgfilename, textfilename, labelfilename,
                 train = True, supervise = True, traintestproportion = 0.667,
                 superviseunsuperviseproportion = [7, 3]):
        super(MyDataset, self).__init__()
        self.imgfilenamerecord = imgfilenamerecord
        self.imgfilename = imgfilename
        self.textfilename = textfilename
        self.labelfilename = labelfilename
        self.modelstrain = train
        self.val = None
        self.test = None
        self.supervise = supervise
        self.pro1 = traintestproportion
        self.pro2 = superviseunsuperviseproportion


        fr = open(self.imgfilenamerecord,'rb')
        self.imgrecordlist = pickle.load(fr)
        for i in range(len(self.imgrecordlist)):
            self.imgrecordlist[i] = self.imgfilename + self.imgrecordlist[i]
        self.imgrecordlist = np.array(self.imgrecordlist)
        self.textlist = np.load(self.textfilename)
        self.labellist = np.load(self.labelfilename)

        
        '''
        upset data.
        '''

        train_supervise = np.load('data/train_index.npy')
        self.train_superviseimgrecordlist = [self.imgrecordlist[i] for i in train_supervise]
        self.train_supervisetextlist = [self.textlist[i] for i in train_supervise]
        self.train_superviselabellist = [self.labellist[i] for i in train_supervise]

        train_unsupervise = np.load('data/unsupervise_index.npy')
        self.unsuperviseimgrecordlist = [self.imgrecordlist[i] for i in train_unsupervise]
        self.unsupervisetextlist = [self.textlist[i] for i in train_unsupervise]
        self.unsuperviselabellist = [self.labellist[i] for i in train_unsupervise]

        val_index = np.load('data/val_index.npy')
        self.val_imgrecordlist = [self.imgrecordlist[i] for i in val_index]
        self.val_textlist = [self.textlist[i] for i in val_index]
        self.val_labellist = [self.labellist[i] for i in val_index]

        test_index = np.load('data/test_index.npy')
        self.testimgrecordlist = [self.imgrecordlist[i] for i in test_index]
        self.testtextlist = [self.textlist[i] for i in test_index]
        self.testlabellist = [self.labellist[i] for i in test_index]


    def supervise_(self):
        self.train = True
        self.supervise = True
        return self

    def val_(self):
        self.test = False
        self.val = True
        self.train = False
        return self

    def test_(self):
        self.test = True
        self.val = False
        self.train = False
        return self

    def unsupervise_(self):
        self.train = True
        self.supervise = False
        return self

    def __getitem__(self, index):
        if self.train == True and self.supervise == True:
            img = Image.open(self.train_superviseimgrecordlist[index]).convert('RGB').resize((256, 256))
            text = self.train_supervisetextlist[index]
            label = self.train_superviselabellist[index]
            img = T.ToTensor()(img)
            text = paddle.to_tensor(text, dtype='float32')
            label = paddle.to_tensor(label, dtype='float32')
            feature = []
            feature.append(img)
            feature.append(text)
            return feature, label
        elif self.train == True and self.supervise == False:
            supervise_img = []
            supervise_text = []
            supervise_label = []
            for i in range(index*self.pro2[0],(index+1)*self.pro2[0]):
                temp_img = Image.open(self.train_superviseimgrecordlist[i]).convert('RGB').resize((256, 256))
                temp_text = self.train_supervisetextlist[i]
                temp_label = self.train_superviselabellist[i]
                temp_img = T.ToTensor()(temp_img)
                temp_text = paddle.to_tensor(temp_text, dtype='float32')
                temp_label = paddle.to_tensor(temp_label, dtype='float32')
                supervise_img.append(temp_img)
                supervise_text.append(temp_text)
                supervise_label.append(temp_label)
            unsupervise_img = []
            unsupervise_text = []
            unsupervise_label = []
            for i in range(index*self.pro2[1],(index+1)*self.pro2[1]):
                temp_img = Image.open(self.unsuperviseimgrecordlist[i]).convert('RGB').resize((256, 256))
                temp_text = self.unsupervisetextlist[i]
                temp_label = self.unsuperviselabellist[i]
                temp_img = T.ToTensor()(temp_img)
                temp_text = paddle.to_tensor(temp_text, dtype='float32')
                temp_label = paddle.to_tensor(temp_label, dtype='float32')
                unsupervise_img.append(temp_img)
                unsupervise_text.append(temp_text)
                unsupervise_label.append(temp_label)
            feature = []
            feature.append(supervise_img)
            feature.append(supervise_text)
            feature.append(unsupervise_img)
            feature.append(unsupervise_text)
            return feature, supervise_label
        elif self.train == False and self.test == True and self.val == False:
            img = Image.open(self.testimgrecordlist[index]).convert('RGB').resize((256, 256))
            text = self.testtextlist[index]
            label = self.testlabellist[index]
            img = T.ToTensor()(img)
            text = paddle.to_tensor(text, dtype='float32')
            label = paddle.to_tensor(label, dtype='float32')
            feature = []
            feature.append(img)
            feature.append(text)
            return feature, label
        elif self.train == False and self.val == True and self.test == False:
            img = Image.open(self.val_imgrecordlist[index]).convert('RGB').resize((256, 256))
            text = self.val_textlist[index]
            label = self.val_labellist[index]
            img = T.ToTensor()(img)
            text = paddle.to_tensor(text, dtype='float32')
            label = paddle.to_tensor(label, dtype='float32')
            feature = []
            feature.append(img)
            feature.append(text)
            return feature, label

    def __len__(self):
        if self.train == True and self.supervise == True:
            return len(self.train_superviselabellist)
        elif self.train == True and self.supervise == False:
            return int(len(self.unsuperviselabellist) / self.pro2[1])
        elif self.train == False and self.test == True:
            return len(self.testlabellist)
        elif self.train == False and self.val == True:
            return len(self.val_labellist)