# -*- coding: utf-8 -*-

import numpy as np
from Model import measure_average_precision
from Model import measure_coverage
from Model import measure_example_auc
from Model import measure_macro_auc
from Model import measure_micro_auc
from Model import measure_ranking_loss

import paddle

def test(Textfeaturemodel, Imgpredictmodel, Textpredictmodel, Imgmodel,
         Predictmodel, Attentionmodel, testdataset, batchsize = 32, cuda = False):

    Textfeaturemodel.eval()
    Imgpredictmodel.eval()
    Textpredictmodel.eval()
    Imgmodel.eval()
    Predictmodel.eval()
    Attentionmodel.eval()

    print('test data:')
    data_loader = paddle.io.DataLoader(dataset=testdataset, batch_size=batchsize, shuffle=False)
    total_predict = []
    img_predict = []
    text_predict = []
    truth = []
    for batch_index, (x, y) in enumerate(data_loader(), 1):
        img_xx = x[0]
        text_xx = x[1]
        label = y.numpy()

        img_xx = img_xx.cuda() if cuda else img_xx
        text_xx = text_xx.cuda() if cuda else text_xx
        imghidden = Imgmodel(img_xx)
        texthidden = Textfeaturemodel(text_xx)

        imgk = Attentionmodel(imghidden)
        textk = Attentionmodel(texthidden)
        modality_attention = []
        modality_attention.append(imgk)
        modality_attention.append(textk)
        modality_attention = paddle.concat(modality_attention, 1)
        softmax = paddle.nn.Softmax(axis=1)
        modality_attention = softmax(modality_attention)
        img_attention = paddle.zeros(shape=[1, len(y)])
        img_attention[0] = modality_attention[:,0]
        img_attention = paddle.t(img_attention)
        text_attention = paddle.zeros(shape=[1, len(y)])
        text_attention[0] = modality_attention[:,1]
        text_attention = paddle.t(text_attention)
        if cuda:
            img_attention = img_attention.cuda()
            text_attention = text_attention.cuda()
        imgpredict = Imgpredictmodel(imghidden)
        textpredict = Textpredictmodel(texthidden)
        feature_hidden = img_attention * imghidden + text_attention * texthidden
        predict = Predictmodel(feature_hidden)

        img_ = imgpredict.cpu().numpy()
        text_ = textpredict.cpu().numpy()
        predict = predict.cpu().numpy()
        total_predict.append(predict)
        img_predict.append(img_)
        text_predict.append(text_)
        truth.append(label)


    total_predict = np.array(total_predict)
    img_predict = np.array(img_predict)
    text_predict = np.array(text_predict)
    truth = np.array(truth)
    temp = total_predict[0]
    for i in range(1, len(total_predict)):
        temp = np.vstack((temp, total_predict[i]))
    total_predict = temp
    temp = img_predict[0]
    for i in range(1, len(img_predict)):
        temp = np.vstack((temp, img_predict[i]))
    img_predict = temp
    temp = text_predict[0]
    for i in range(1, len(text_predict)):
        temp = np.vstack((temp, text_predict[i]))
    text_predict = temp
    temp = truth[0]
    for i in range(1, len(truth)):
        temp = np.vstack((temp, truth[i]))
    truth = temp
    average_precison1 = measure_average_precision.average_precision(total_predict, truth)
    average_precison2 = measure_average_precision.average_precision(img_predict, truth)
    average_precison3 = measure_average_precision.average_precision(text_predict, truth)
    coverage1 = measure_coverage.coverage(total_predict, truth)
    coverage2 = measure_coverage.coverage(img_predict, truth)
    coverage3 = measure_coverage.coverage(text_predict, truth)

    example_auc1 = measure_example_auc.example_auc(total_predict, truth)
    example_auc2 = measure_example_auc.example_auc(img_predict, truth)
    example_auc3 = measure_example_auc.example_auc(text_predict, truth)

    macro_auc1 = measure_macro_auc.macro_auc(total_predict, truth)
    macro_auc2 = measure_macro_auc.macro_auc(img_predict, truth)
    macro_auc3 = measure_macro_auc.macro_auc(text_predict, truth)

    micro_auc1 = measure_micro_auc.micro_auc(total_predict, truth)
    micro_auc2 = measure_micro_auc.micro_auc(img_predict, truth)
    micro_auc3 = measure_micro_auc.micro_auc(text_predict, truth)

    ranking_loss1 = measure_ranking_loss.ranking_loss(total_predict, truth)
    ranking_loss2 = measure_ranking_loss.ranking_loss(img_predict, truth)
    ranking_loss3 = measure_ranking_loss.ranking_loss(text_predict, truth)

    return average_precison1, average_precison2, average_precison3, coverage1, coverage2, coverage3, example_auc1, example_auc2, example_auc3, macro_auc1, macro_auc2, macro_auc3, micro_auc1, micro_auc2, micro_auc3, ranking_loss1, ranking_loss2, ranking_loss3


def texttest(Textfeaturemodel, Textpredictmodel, testdataset, batchsize = 32, cuda = False):

    Textfeaturemodel.eval()
    Textpredictmodel.eval()

    print('test text data:')
    data_loader = paddle.io.DataLoader(dataset = testdataset, batch_size = batchsize,
                                       shuffle = False, num_workers = 0)
    text_predict = []
    truth = []
    for batch_index, (x, y) in enumerate(data_loader(), 1):
        text_xx = x[1]
        label = y.numpy()
        text_xx = text_xx.cuda() if cuda else text_xx
        textxx = Textfeaturemodel(text_xx)
        textyy = Textpredictmodel(textxx)
        text_ = textyy.cpu().numpy()
        text_predict.append(text_)
        truth.append(label)

    text_predict = np.array(text_predict)
    truth = np.array(truth)
    temp = text_predict[0]
    for i in range(1, len(text_predict)):
        temp = np.vstack((temp, text_predict[i]))
    text_predict = temp
    temp = truth[0]
    for i in range(1, len(truth)):
        temp = np.vstack((temp, truth[i]))
    truth = temp
    average_precison = measure_average_precision.average_precision(text_predict, truth)
    return average_precison


def Imgtest(Imgmodel, Imgpredictmodel, testdataset, batchsize = 32, cuda = False):

    Imgmodel.eval()
    Imgpredictmodel.eval()

    print('test img data:')
    data_loader = paddle.io.DataLoader(dataset = testdataset, batch_size = batchsize,
                                       shuffle = False, num_workers = 0)
    img_predict = []
    truth = []
    for batch_index, (x, y) in enumerate(data_loader(), 1):
        img_xx = x[0]
        label = y.numpy()

        img_xx = img_xx.cuda() if cuda else img_xx
        imgxx = Imgmodel(img_xx)
        imgyy = Imgpredictmodel(imgxx)
        img_ = imgyy.cpu().numpy()
        img_predict.append(img_)
        truth.append(label)

    img_predict = np.array(img_predict)
    truth = np.array(truth)
    temp = img_predict[0]
    for i in range(1, len(img_predict)):
        temp = np.vstack((temp, img_predict[i]))
    img_predict = temp
    temp = truth[0]
    for i in range(1, len(truth)):
        temp = np.vstack((temp, truth[i]))
    truth = temp
    average_precison = measure_average_precision.average_precision(img_predict, truth)
    #example_f1 = measure_example_f1.example_f1(text_predict, truth)
    return average_precison

