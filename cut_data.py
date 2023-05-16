# -*- coding: utf-8 -*-
import os
import shutil
import random

# 数据集所在文件夹路径
data_dir = 'D:\\PycharmProjects\\dogs-vs-cats\\train'

# 训练集和测试集所在文件夹路径
train_dir = 'D:\\PycharmProjects\\DogsVsCats\\data\\train'
test_dir = 'D:\\PycharmProjects\\DogsVsCats\\data\\test'

# 用于创建训练集和测试集文件夹
if not os.path.exists(train_dir):
    os.makedirs(train_dir)

if not os.path.exists(test_dir):
    os.makedirs(test_dir)

# 类别名列表
classes = ['cat', 'dog']

# 每个类别需要的样本数量
samples_per_class = {'cat': 2000, 'dog': 2000}

# 分层抽样
for class_name in classes:
    # 获取该类别的所有文件
    files = os.listdir(os.path.join(data_dir, class_name))
    # 随机打乱文件顺序
    random.shuffle(files)
    # 计算训练集和测试集中该类别需要的样本数量
    train_samples = int(samples_per_class[class_name] * 0.8)
    test_samples = samples_per_class[class_name] - train_samples
    # 复制文件到训练集文件夹
    for i in range(train_samples):
        file_name = files[i]
        src = os.path.join(data_dir, class_name, file_name)
        dst = os.path.join(train_dir, f'{class_name}.{i}.jpg')
        shutil.copyfile(src, dst)
    # 复制文件到测试集文件夹
    for i in range(train_samples, train_samples + test_samples):
        file_name = files[i]
        src = os.path.join(data_dir, class_name, file_name)
        dst = os.path.join(test_dir, f'{class_name}.{i}.jpg')
        shutil.copyfile(src, dst)
