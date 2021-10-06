# -*- coding: utf-8 -*-

import numpy as np
from Model import Test
import datetime
import paddle

import logging


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

cita = 1.003
loss_batch = 20

def train(Textfeaturemodel, Imgpredictmodel, Textpredictmodel, Imgmodel, Attentionmodel, Predictmodel, dataset,
          supervise_epochs = 20, text_supervise_epochs = 50, img_supervise_epochs = 50,
          lr_supervise = 0.01, text_lr_supervise = 0.0001, img_lr_supervise = 0.0001,
          weight_decay = 0, batchsize = 32,lambda1=0.01,lambda2=1, textbatchsize = 32, imgbatchsize = 32, cuda = False, savepath = ''):

    logger = get_logger('exp.log')
    logger.info('start training!')

    '''
    pretrain ImgNet
    '''
    Imgmodel.train()
    Imgpredictmodel.train()

    scheduler = paddle.optimizer.lr.StepDecay(learning_rate=img_lr_supervise, step_size=500, gamma=0.9)

    optims_Imgmodel = paddle.optimizer.Adam(learning_rate=img_lr_supervise,
                                      parameters=Imgmodel.parameters())
    optims_Imgpredictmodel = paddle.optimizer.Adam(learning_rate=img_lr_supervise,
                                      parameters=Imgpredictmodel.parameters())

    criterion = paddle.nn.BCELoss()
    train_img_supervise_loss = []
    batch_count = 0
    loss = 0

    for epoch in range(1, img_supervise_epochs + 1):
        print('train img supervise data:', epoch)
        data_loader = paddle.io.DataLoader(dataset = dataset.supervise_(), batch_size = imgbatchsize,
                                           shuffle = True, num_workers = 0)

        for batch_index, (x, y) in enumerate(data_loader(), 1):
            batch_count += 1

            img_xx = x[0]
            label = y
            img_xx = img_xx.cuda() if cuda else img_xx
            label = label.cuda() if cuda else label
            imgxx = Imgmodel(img_xx)
            imgyy = Imgpredictmodel(imgxx)
            img_supervise_batch_loss = criterion(imgyy, label)
            loss += img_supervise_batch_loss.item()
            if batch_count >= loss_batch:
                loss = loss/loss_batch
                train_img_supervise_loss.append(loss)
                loss = 0
                batch_count = 0

            img_supervise_batch_loss.backward()
            optims_Imgmodel.step()
            optims_Imgpredictmodel.step()
            optims_Imgmodel.clear_grad()
            optims_Imgpredictmodel.clear_grad()
            scheduler.step()

        if epoch % (img_supervise_epochs) == 0:
            filename = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
            paddle.save(Imgmodel.state_dict(), savepath + filename + 'pretrainimgfeature.pdparams')
            paddle.save(Imgpredictmodel.state_dict(), savepath + filename + 'pretrainimgpredict.pdparams')
            np.save(savepath + filename + "imgsuperviseloss.npy", train_img_supervise_loss)
            acc = Test.Imgtest(Imgmodel, Imgpredictmodel, dataset.test_(), batchsize = imgbatchsize, cuda = cuda)
            print('img supervise', epoch, acc)
            np.save(savepath + filename + "imgsuperviseacc.npy", [acc])
    

    '''
    pretrain TextNet.
    ''' 

    Textfeaturemodel.train()
    Textpredictmodel.train()

    scheduler = paddle.optimizer.lr.StepDecay(learning_rate=text_lr_supervise, step_size=500, gamma=0.9)
    optims_Textfeaturemodel = paddle.optimizer.Adam(learning_rate=text_lr_supervise,
                                      parameters=Textfeaturemodel.parameters())
    optims_Textpredictmodel = paddle.optimizer.Adam(learning_rate=text_lr_supervise,
                                      parameters=Textpredictmodel.parameters())
    criterion = paddle.nn.BCELoss()
    train_text_supervise_loss = []
    batch_count = 0
    loss = 0

    for epoch in range(1, text_supervise_epochs + 1):
        print('train text supervise data:', epoch)
        data_loader = paddle.io.DataLoader(dataset=dataset.supervise_(), batch_size=textbatchsize,
                                      shuffle=True,
                                      num_workers=0)
        for batch_index, (x, y) in enumerate(data_loader(), 1):
            batch_count += 1

            text_xx = x[1]
            label = y

            text_xx = text_xx.cuda() if cuda else text_xx
            label = label.cuda() if cuda else label
            textxx = Textfeaturemodel(text_xx)
            textyy = Textpredictmodel(textxx)
            text_supervise_batch_loss = criterion(textyy, label)
            loss += text_supervise_batch_loss.item()
            if batch_count >= loss_batch:
                loss = loss/loss_batch
                train_text_supervise_loss.append(loss)
                loss = 0
                batch_count = 0

            text_supervise_batch_loss.backward()
            optims_Textfeaturemodel.step()
            optims_Textpredictmodel.step()
            optims_Textfeaturemodel.clear_grad()
            optims_Textpredictmodel.clear_grad()
            scheduler.step()


        filename = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
        paddle.save(Textfeaturemodel.state_dict(), savepath + filename + 'pretraintextfeature.pdparams')
        paddle.save(Textpredictmodel.state_dict(), savepath + filename + 'pretraintextpredict.pdparams')
        np.save(savepath + filename + "textsuperviseloss.npy", train_text_supervise_loss)
        acc = Test.texttest(Textfeaturemodel, Textpredictmodel, dataset.test_(), batchsize = textbatchsize, cuda = cuda)

        logger.info('Epoch:[{}/{}]\n'
                    'acc={:.3f}'.format(epoch, text_supervise_epochs + 1, acc))
    
    '''
    train data mode.
    '''   

    Textfeaturemodel.train()
    Imgpredictmodel.train()
    Textpredictmodel.train()        
    Attentionmodel.train()
    Imgmodel.train()
    Predictmodel.train()

    optims_Attentionmodel = paddle.optimizer.Adam(learning_rate=lr_supervise,
                                      parameters=Attentionmodel.parameters())
    optims_Textfeaturemodel = paddle.optimizer.Adam(learning_rate=lr_supervise,
                                      parameters=Textfeaturemodel.parameters())
    optims_Imgpredictmodel = paddle.optimizer.Adam(learning_rate=lr_supervise,
                                      parameters=Imgpredictmodel.parameters())
    optims_Textpredictmodel = paddle.optimizer.Adam(learning_rate=lr_supervise,
                                      parameters=Textpredictmodel.parameters())
    optims_Predictmodel = paddle.optimizer.Adam(learning_rate=lr_supervise,
                                                parameters=Predictmodel.parameters())
    optims_Imgmodel = paddle.optimizer.Adam(learning_rate=lr_supervise,
                                                parameters=Imgmodel.parameters())

    criterion = paddle.nn.BCELoss()
    train_supervise_loss = []
    batch_count = 0
    loss = 0
    best_result = 10

    for epoch in range(1, supervise_epochs + 1):
        print('train supervise data:', epoch)
        data_loader = paddle.io.DataLoader(dataset = dataset.unsupervise_(), batch_size = batchsize,
                                           shuffle = True, num_workers = 0)
        for batch_index, (x, y) in enumerate(data_loader(), 1):
            batch_count += 1

            x[0] = paddle.concat(x[0], 0)
            x[1] = paddle.concat(x[1], 0)
            y = paddle.concat(y, 0)

            '''
            Attention architecture and use bceloss.
            '''
            supervise_img_xx = x[0]
            supervise_text_xx = x[1]
            label = y

            supervise_img_xx = supervise_img_xx.cuda() if cuda else supervise_img_xx
            supervise_text_xx = supervise_text_xx.cuda() if cuda else supervise_text_xx
            label = label.cuda() if cuda else label
            supervise_imghidden = Imgmodel(supervise_img_xx)
            supervise_texthidden = Textfeaturemodel(supervise_text_xx)

            supervise_imgpredict = Imgpredictmodel(supervise_imghidden)
            supervise_textpredict = Textpredictmodel(supervise_texthidden)
            
            supervise_imgk = Attentionmodel(supervise_imghidden)
            supervise_textk = Attentionmodel(supervise_texthidden)
            modality_attention = []
            modality_attention.append(supervise_imgk)
            modality_attention.append(supervise_textk)

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
            supervise_feature_hidden = img_attention * supervise_imghidden + text_attention * supervise_texthidden
            supervise_predict = Predictmodel(supervise_feature_hidden)

            totalloss = criterion(supervise_predict, label)
            imgloss = criterion(supervise_imgpredict, label)
            textloss = criterion(supervise_textpredict, label)
            '''
            Diversity Measure code.
            '''
            similar = paddle.bmm(supervise_imgpredict.unsqueeze(1),
                                 supervise_textpredict.unsqueeze(2))

            similar = paddle.reshape(similar, shape=[supervise_imgpredict.shape[0]])
            norm_matrix_img = paddle.norm(supervise_imgpredict, p=2, axis=1)
            norm_matrix_text = paddle.norm(supervise_textpredict, p=2, axis=1)
            div = paddle.mean(similar/(norm_matrix_img * norm_matrix_text))

            supervise_loss = imgloss + textloss + totalloss*2
            '''
            Robust Consistency Measure code.
            '''
            x[2] = paddle.concat(x[2], 0)
            x[3] = paddle.concat(x[3], 0)
            unsupervise_img_xx = x[2]
            unsupervise_text_xx = x[3]

            unsupervise_img_xx = unsupervise_img_xx.cuda() if cuda else unsupervise_img_xx
            unsupervise_text_xx = unsupervise_text_xx.cuda() if cuda else unsupervise_text_xx

            unsupervise_imghidden = Imgmodel(unsupervise_img_xx)
            unsupervise_texthidden = Textfeaturemodel(unsupervise_text_xx)
            unsupervise_imgpredict = Imgpredictmodel(unsupervise_imghidden)
            unsupervise_textpredict = Textpredictmodel(unsupervise_texthidden)

            unsimilar = paddle.bmm(unsupervise_imgpredict.unsqueeze(1),
                                 unsupervise_textpredict.unsqueeze(2))

            unsimilar = paddle.reshape(unsimilar, shape=[unsupervise_imgpredict.shape[0]])

            unnorm_matrix_img = paddle.norm(unsupervise_imgpredict, p=2, axis = 1)
            unnorm_matrix_text = paddle.norm(unsupervise_textpredict, p=2, axis = 1)

            dis = 2 - unsimilar/(unnorm_matrix_img * unnorm_matrix_text)

            mask_1 = paddle.abs(dis) < cita
            tensor1 = paddle.masked_select(dis, mask_1)
            mask_2 = paddle.abs(dis) >= cita
            tensor2 = paddle.masked_select(dis, mask_2)
            tensor1loss = paddle.sum(tensor1 * tensor1/2)
            tensor2loss = paddle.sum(cita * (paddle.abs(tensor2) - 1/2 * cita))
            unsupervise_loss = (tensor1loss + tensor2loss)/unsupervise_img_xx.shape[0]
            total_loss = supervise_loss + 0.01* div +  unsupervise_loss
            
            loss += total_loss.item()
            if batch_count >= loss_batch:
                loss = loss/loss_batch
                train_supervise_loss.append(loss)
                loss = 0
                batch_count = 0

            total_loss.backward()
            optims_Attentionmodel.step()
            optims_Textfeaturemodel.step()
            optims_Imgpredictmodel.step()
            optims_Imgmodel.step()
            optims_Textpredictmodel.step()
            optims_Predictmodel.step()

            optims_Attentionmodel.clear_grad()
            optims_Textfeaturemodel.clear_grad()
            optims_Imgpredictmodel.clear_grad()
            optims_Imgmodel.clear_grad()
            optims_Textpredictmodel.clear_grad()
            optims_Predictmodel.clear_grad()

            scheduler.step()

        if epoch % 1 == 0:

            print("val: ", len(dataset.val_()))
            acc1, acc2, acc3, coverage1, coverage2, coverage3, example_auc1, example_auc2, \
            example_auc3, macro_auc1, macro_auc2, macro_auc3, micro_auc1, micro_auc2, micro_auc3, \
            ranking_loss1, ranking_loss2, ranking_loss3 = Test.test(Textfeaturemodel, Imgpredictmodel,
                                                                    Textpredictmodel, Imgmodel,
                                                                    Predictmodel, Attentionmodel,
                                                                    dataset.val_(),
                                                                    batchsize = batchsize,
                                                                    cuda = cuda)

            logger.info('Epoch:[{}/{}]\n'
                        'acc1={:.5f}, acc2={:.5f}, acc3={:.5f}\n'
                        'coverage1={:.5f}, coverage2={:.5f}, coverage3={:.5f}\n'
                        'example_auc1={:.5f}, example_auc2={:.5f}, example_auc3={:.5f}\n'
                        'macro_auc1={:.5f}, macro_auc2={:.5f}, macro_auc3={:.5f}\n'
                        'micro_auc1={:.5f}, micro_auc2={:.5f}, micro_auc3={:.5f}\n'
                        'ranking_loss1={:.5f}, ranking_loss2={:.5f}, ranking_loss3={:.5f}'
                        .format(epoch, supervise_epochs + 1, acc1, acc2, acc3,
                                coverage1, coverage2, coverage3, example_auc1, example_auc2, example_auc3,
                                macro_auc1, macro_auc2, macro_auc3, micro_auc1, micro_auc2, micro_auc3,
                                ranking_loss1, ranking_loss2, ranking_loss3))

            paddle.save(Textfeaturemodel.state_dict(), savepath + 'Textfeaturemodel.pdparams')
            paddle.save(Imgpredictmodel.state_dict(), savepath + 'Imgpredictmodel.pdparams')
            paddle.save(Textpredictmodel.state_dict(), savepath + 'Textpredictmodel.pdparams')
            paddle.save(Imgmodel.state_dict(), savepath + 'Imgmodel.pdparams')
            paddle.save(Predictmodel.state_dict(), savepath + 'Predictmodel.pdparams')
            paddle.save(Attentionmodel.state_dict(), savepath + 'Attentionmodel.pdparams')

            if coverage1 < best_result:
                best_result = coverage1
                print("coverage1 : ", coverage1)
                paddle.save(Textfeaturemodel.state_dict(), savepath + 'Textfeaturemodel-best.pdparams')
                paddle.save(Imgpredictmodel.state_dict(), savepath + 'Imgpredictmodel-best.pdparams')
                paddle.save(Textpredictmodel.state_dict(), savepath + 'Textpredictmodel-best.pdparams')
                paddle.save(Imgmodel.state_dict(), savepath + 'Imgmodel-best.pdparams')
                paddle.save(Predictmodel.state_dict(), savepath + 'Predictmodel-best.pdparams')
                paddle.save(Attentionmodel.state_dict(), savepath + 'Attentionmodel-best.pdparams')



    Textfeaturemodel.eval()
    Textfeaturemodel.set_state_dict(paddle.load(savepath + 'Textfeaturemodel-best.pdparams'))
    Imgpredictmodel.eval()
    Imgpredictmodel.set_state_dict(paddle.load(savepath + 'Imgpredictmodel-best.pdparams'))
    Textpredictmodel.eval()
    Textpredictmodel.set_state_dict(paddle.load(savepath + 'Textpredictmodel-best.pdparams'))
    Imgmodel.eval()
    Imgmodel.set_state_dict(paddle.load(savepath + 'Imgmodel-best.pdparams'))
    Predictmodel.eval()
    Predictmodel.set_state_dict(paddle.load(savepath + 'Predictmodel-best.pdparams'))
    Attentionmodel.eval()
    Attentionmodel.set_state_dict(paddle.load(savepath + 'Attentionmodel-best.pdparams'))


    print("test : ", len(dataset.test_()))
    acc1, acc2, acc3, coverage1, coverage2, coverage3, example_auc1, example_auc2, \
    example_auc3, macro_auc1, macro_auc2, macro_auc3, micro_auc1, micro_auc2, micro_auc3, \
    ranking_loss1, ranking_loss2, ranking_loss3 = Test.test(Textfeaturemodel, Imgpredictmodel,
                                                            Textpredictmodel, Imgmodel, Predictmodel,
                                                            Attentionmodel, dataset.test_(), batchsize=batchsize,
                                                            cuda=cuda)

    logger.info('Test: \n'
                'acc1={:.5f}\n'
                'coverage1={:.5f}\n'
                'example_auc1={:.5f}\n'
                'macro_auc1={:.5f}\n'
                'micro_auc1={:.5f}\n'
                'ranking_loss1={:.5f}'
                .format(acc1, coverage1, example_auc1, macro_auc1, micro_auc1, ranking_loss1))

    logger.info('finish training!')

    return train_supervise_loss
