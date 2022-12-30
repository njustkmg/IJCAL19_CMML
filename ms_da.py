import os
from mindspore import dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset.transforms.c_transforms as C2


def create_dataset(dataset_path, batch_size=3, rank_size=1, rank_id=0, do_train=True):
    """
    create a train or eval imagenet2012 dataset for resnet50

    Args:
        dataset_path(string): the path of dataset.
        batch_size(int): the batch size of dataset. Default: 32
        rank_size(int): total num of devices for training. Default: 1,
                        greater than 1 in distributed training
        rank_id(int): logical sequence in all devices. Default: 1,
                      can be greater than i in distributed training

    Returns:
        dataset
    """
    data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=1, shuffle=do_train,
                                     num_shards=rank_size, shard_id=rank_id)

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    # define map operations
    trans = [
        C.Decode(),
        C.Resize([224, 224]),
        # C.CenterCrop(224),
        C.Normalize(mean=mean, std=std),
        C.HWC2CHW()
    ]

    type_cast_op = C2.TypeCast(mstype.int32)

    data_set = data_set.map(operations=trans, input_columns="image", num_parallel_workers=1)
    data_set = data_set.map(operations=type_cast_op, input_columns="label", num_parallel_workers=1)

    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=do_train)

    return data_set