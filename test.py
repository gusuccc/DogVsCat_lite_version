# -*- coding: utf-8 -*-
from getdata import DogsVSCatsDataset as DVCD
from network import Net
import torch
import torch.nn as nn
from PIL import Image
import getdata
import torch.nn.functional as F

dataset_dir = './data'  # 数据集路径
model_file = './model/model_1.pth'  # 模型保存路径


def test():
    datafile = DVCD('test', dataset_dir)  # 实例化一个数据集

    print('Dataset loaded! length of train set is {0}'.format(len(datafile)))

    model = Net()  # 实例化一个网络
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_file))  # 加载训练好的模型参数
    model.eval()  # 设定为评估模式，即计算过程中不要dropout
    # get data
    imgs = []  # img
    imgs_data = []  # img data
    total_images = 0
    total_correct = 0
    img_list = datafile.list_img
    img_label = datafile.list_label

    for img_dir in img_list:
        img = Image.open(img_dir)  # 打开图像
        img_data = getdata.dataTransform(img)  # 转换成torch tensor数据

        imgs.append(img)  # 图像list
        imgs_data.append(img_data)  # tensor list
    imgs_data = torch.stack(imgs_data)  # tensor list合成一个4D tensor
    # calculation
    out = model(imgs_data)  # 对每个图像进行网络计算
    out = F.softmax(out, dim=1)  # 输出概率化
    out = out.data.cpu().numpy()  # 转成numpy数据
    # 猫0狗1
    for idx in range(datafile.data_size):
        label = img_label[idx]
        predicted = (out[idx, label] > 0.5)
        total_correct += predicted
        #print('label: %d, predict: %d' % (label, predicted))


    total_images=datafile.data_size
    accuracy = 100 * total_correct / total_images
    print('frame: %d, correct: %d' % (total_images, total_correct))
    print('Test Accuracy: {0:.2f}%'.format(accuracy))


if __name__ == '__main__':
    test()
