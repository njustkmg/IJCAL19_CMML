# mindspore related
import argparse
from mindspore import context
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from ms_model import *
from ms_da import *
import numpy
from mindspore import Model
from mindvision.engine.callback import LossMonitor

# 设置模型保存参数
# config_ck = CheckpointConfig(save_checkpoint_steps=1875, keep_checkpoint_max=10)

# 应用模型保存参数
# ckpoint = ModelCheckpoint(prefix="Ysnaker", directory="./ms_net", config=config_ck)

def train():
    # read all the data
    train_data = create_dataset('./final_ms/train', batch_size=1)
    test_data = create_dataset('./final_ms/test', batch_size=1)

    start_result = 0
    batch_size = 3
    # ready for the model
    net = Net()
    expend = mindspore.ops.ExpandDims()
    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    optimizer = nn.Adam(list(net.trainable_params()), learning_rate=1e-5)
    model_with_criterion = nn.WithLossCell(net, criterion)
    train_model = nn.TrainOneStepCell(model_with_criterion, optimizer)
    train_model.set_train()
    # begin to train
    steps = train_data.get_dataset_size()

    for epoch in range(100):
        step = 0
        for batch_idx, (batch_identify_x, batch_identify_y) in enumerate(train_data):
            data_identify, label_identify = batch_identify_x, batch_identify_y
            label = mindspore.Tensor(label_identify, mindspore.int32)
            # depth, on_value, off_value = 2, mindspore.Tensor(1.0, mindspore.float32), mindspore.Tensor(0.0, mindspore.float32)
            # onehot = ops.OneHot()
            # label = onehot(label, depth, on_value, off_value)
            label = mindspore.Tensor(label, mindspore.int32)
            loss = train_model(data_identify, label)
            step += 1

            if step % 200 == 0:
                mindspore.save_checkpoint(net, './ms_net/snaker.ckpt')
                print(f"Epoch: [{epoch} / 100], "f"batch_id: [{batch_idx}] " f"step: [{step} / {steps}], "f"loss: {loss}")
                nets = Net()
                param_dict = mindspore.load_checkpoint("./ms_net/snaker.ckpt")
                acc = nn.Accuracy()
                F1 = nn.F1()
                F1.clear()
                acc.clear()
                # 重新定义一个LeNet神经网络
                # 将参数加载到网络中
                mindspore.load_param_into_net(nets, param_dict)
                eval_net = nn.WithEvalCell(nets, criterion)
                eval_net.set_train(False)
                for data in test_data.create_dict_iterator():
                    outputs = eval_net(data["image"], data["label"])
                    F1.update(outputs[1], outputs[2])
                    acc.update(outputs[1], outputs[2])
                    # losss.update(outputs[0])
                print('F1 :' + str(F1.eval()))
                print('ACC :' + str(acc.eval()))

    # model = Model(network=model, loss_fn=criterion, optimizer=optimizer, metrics={"F1": F1(), "Acc": Accuracy()})
    # model.train(3, train_data, callbacks=[LossMonitor(1e-5), ckpoint])
    # eval_result = model.eval(test_data)
    # print(eval_result)
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Sneaker model')
    parser.add_argument('--device_target', type=str, default="GPU", choices=['Ascend', 'GPU', 'CPU'])
    params = parser.parse_known_args()[0]
    context.set_context(mode=context.GRAPH_MODE, device_target=params.device_target)
    train()



