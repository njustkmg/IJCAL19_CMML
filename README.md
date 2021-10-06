# CMML-Paddle

## 一、简介

- 本项目基于 Paddle 框架复现论文 Comprehensive Semi-Supervised Multi-Modal Learning
- 论文：[Comprehensive Semi-Supervised Multi-Modal Learning | IJCAI](https://www.ijcai.org/proceedings/2019/568)
- 参考实现：https://github.com/njustkmg/IJCAI19_CMML

## 二、数据集

本项目数据集使用 MSCOCO, 数据集划分根据原始论文和主办方规定

具体地，将原始的数据集处理成以下文件：

- 标签文件，如 COCO_label.npy
- 文本特征文件(使用 BOW 提取特征)，如 COCO_text.npy
- 图片索引，如 COCO_image.pkl
- 图片文件夹，保存图片原始文件


## 四、依赖环境

```
numpy                     1.19.5                   
openssl                   1.1.1k             
paddlepaddle-gpu          2.1.2.post110            
Pillow                    8.3.1                    
pip                       21.2.2          
protobuf                  3.17.3                    
python                    3.6.13              
readline                  8.1               
requests                  2.26.0                   
setuptools                52.0.0           
six                       1.16.0                    
sqlite                    3.36.0               
tk                        8.6.10              
urllib3                   1.26.6                    
wheel                     0.37.0          
xz                        5.2.5              
zlib                      1.2.11
```

## 五、代码结构与使用

方法的主要代码存放于 Model 文件夹下：

- Model.py 为 CMML 模型
- Trainer.py 和 Test.py 为模型训练代码
- measure_*.py 为实验使用的多个指标计算代码

~~~
.
├── code
│   ├── Deep_attention_strong_weak_train.py
│   └── Model
│       ├── Data.py
│       ├── measure_average_precision.py
│       ├── measure_coverage.py
│       ├── measure_example_auc.py
│       ├── measure_macro_auc.py
│       ├── measure_micro_auc.py
│       ├── measure_ranking_loss.py
│       ├── Model.py
│       ├── Test.py
│       └── Train.py
├── data
└── models
~~~

程序运行 

```
python Deep_attention_strong_week_train.py 
```

需先定义文件位置（默认存放于data/目录下）：

- textfilename：文本特征文件地址
- imgfilenamerecord：图片索引文件地址
- imgfilename：图片文件夹存放目录
- labelfilename：标签文件地址

实验运行超参数位于 Deep_attention_strong_week_train.py 中，根据具体说明调整。如果仅测试模型，将epoch设置为0即可。

## 六、模型信息

| 信息     | 说明                                                         |
| -------- | ------------------------------------------------------------ |
| 作者     | biubiu13                                                     |
| 时间     | 2021.09                                                      |
| 框架版本 | Paddle 2.1.2                                                 |
| 应用场景 | 多模态分类                                                   |
| 支持硬件 | CPU、GPU                                                     |
| 下载链接 | [预训练模型](https://pan.baidu.com/s/1c4TTg3VkLacMQiJuxHZfjA) 提取码：nsfc |
| 下载链接 | [训练日志](https://pan.baidu.com/s/1ifagsblHLYfUDlFIs58bCg) 提取码：2pqa |




