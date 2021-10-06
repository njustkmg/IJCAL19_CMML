# -*- coding: utf-8 -*-
from argparse import ArgumentParser
from Model import Data
from Model import Model
from Model import Train
import os
import paddle

parser = ArgumentParser('Strong and weak modality Attention-based Deep learning.')
parser.add_argument('--use-gpu', type = bool, default = True)
parser.add_argument('--visible-gpu', type = str, default = '0')
parser.add_argument('--textfilename', type = str, default = 'data/coco_text.npy')#Path of text madality feature data
parser.add_argument('--imgfilenamerecord', type = str, default = 'data/coco_imgs.pkl')#Path of name list of img madality data
parser.add_argument('--imgfilename', type = str, default = 'data/mscoco/')#Path of img madality data
parser.add_argument('--labelfilename', type = str, default = 'data/coco_label.npy')#Path of data label
parser.add_argument('--savepath', type = str, default = 'models/')
parser.add_argument('--textbatchsize', type = int, default = 32)
parser.add_argument('--imgbatchsize', type = int, default = 32)
parser.add_argument('--batchsize', type = int, default = 4)#train and test batchsize
parser.add_argument('--Textfeaturepara', type = str, default = '2912, 256, 128')#architecture of text feature network
parser.add_argument('--Imgpredictpara', type = str, default = '128, 20')#architecture of img predict network
parser.add_argument('--Textpredictpara', type = str, default = '128, 20')#architecture of text predict network
parser.add_argument('--Predictpara', type = str, default = '128, 20')#architecture of attention predict network
parser.add_argument('--Attentionparameter', type = str, default = '128, 64, 32, 1')#architecture of attention network
parser.add_argument('--img-supervise-epochs', type = int, default = 0)
parser.add_argument('--text-supervise-epochs', type = int, default = 1)
parser.add_argument('--epochs', type = int, default = 20)# train epochs
parser.add_argument('--img-lr-supervise', type = float, default = 0.0001)
parser.add_argument('--text-lr-supervise', type = float, default = 0.0001)
parser.add_argument('--lr-supervise', type = float, default = 0.0001)#train Learning rate
parser.add_argument('--weight-decay', type = float, default = 0)
parser.add_argument('--traintestproportion', type = float, default = 0.667)#ratio of train data to test data
parser.add_argument('--lambda1', type = float, default = 0.01)#ratio of train data to test data
parser.add_argument('--lambda2', type = float, default = 1)#ratio of train data to test data
parser.add_argument('--superviseunsuperviseproportion', type = str, default = '3, 7')#ratio of supervise data to unsupervise data

if __name__ == '__main__':
    args = parser.parse_args()

    '''
    decide whether to use cuda or not.
    '''
    if args.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpu
        cuda = args.use_gpu

    '''
    initialize input vector.
    '''
    param = args.superviseunsuperviseproportion.split(',')
    superviseunsuperviseproportion = list(map(int, param))
    dataset = Data.MyDataset(args.imgfilenamerecord, args.imgfilename, args.textfilename,
                             args.labelfilename, train = True, supervise = True,
                             traintestproportion = args.traintestproportion,
                             superviseunsuperviseproportion = superviseunsuperviseproportion)

    '''
    initialize model.
    '''
    param = args.Textfeaturepara.split(',')
    Textfeatureparam = list(map(int, param))
    param = args.Predictpara.split(',')
    Predictparam = list(map(int, param))
    param = args.Imgpredictpara.split(',')
    Imgpredictparam = list(map(int, param))
    param = args.Textpredictpara.split(',')
    Textpredictparam = list(map(int, param))
    param = args.Attentionparameter.split(',')
    Attentionparam = list(map(int, param))
    # paddle
    Textfeaturemodel, Imgpredictmodel, Textpredictmodel, \
    Imgmodel, Attentionmodel, Predictmodel = Model.generate_model(Textfeatureparam,
                                                                  Imgpredictparam, Textpredictparam,
                                                                  Attentionparam, Predictparam)
    
    '''
    model train.
    '''

    print(args)
    train_supervise_loss = Train.train(Textfeaturemodel, Imgpredictmodel, Textpredictmodel, Imgmodel, Attentionmodel, Predictmodel, dataset,
                                       supervise_epochs = args.epochs, text_supervise_epochs = args.text_supervise_epochs, img_supervise_epochs = args.img_supervise_epochs,
                                       lr_supervise = args.lr_supervise, text_lr_supervise = args.text_lr_supervise, img_lr_supervise = args.img_lr_supervise,
                                       weight_decay = args.weight_decay, batchsize = args.batchsize, textbatchsize = args.textbatchsize, imgbatchsize = args.imgbatchsize, cuda = cuda, savepath = args.savepath,lambda1=args.lambda1,lambda2=args.lambda2)
