from PIL import Image
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
import matplotlib.pyplot as plt
from torchvision import models
# GradCam
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import random

import os
import json

# CPUとGPUのどちらを使うかを指定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ニューラルネットワークの定義
num_classes = 5 # 0~4で分類
net = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
for param in net.parameters():
    param.requires_grad = False
net.classifier = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(256 * 6 * 6, 4096),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096),
    nn.ReLU(inplace=True),
    nn.Linear(4096, num_classes),
)
net = net.to(device)

# 重みの読み込み
net.load_state_dict(torch.load('model_SP.pth'))

# データの前処理を行うtransformを作成
transform = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224), # 画像のランダムな場所を切り抜き、(224, 224)にリサイズ
        transforms.RandomHorizontalFlip(), # 1/2で左右反転
        transforms.ToTensor(), # PILImageを0-1のテンソルに変換
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # 標準化
    ]),
    'test': transforms.Compose([
        transforms.Resize(256), # (256, 256)にリサイズ
        transforms.CenterCrop(224), # 画像の中心に合わせて(224, 224)で切り抜く
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    'display': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
}

# データセットの読み込み
data = []
label = []
soft_label = {}

# jsonファイルから主観評価実験の結果を読み込む
directory = '/home/miyamoto/public_html/exp202309/artificiality_robopitch2/exp-data/logql/'
file_list = os.listdir(directory)
for j in range(len(file_list)):
    f = open(directory + file_list[j], 'r')
    json_directory = json.load(f)
    qlist = json_directory['q']
    alist = json_directory['a']
    for i in range(40):
        robot = qlist[i]['robot']
        data.append(robot)
        ans = alist[i]
        label.append(ans)

# dataの重複をなくし、画像の名前のリストを生成
image_names = list(set(data))

# 画像の名前に対して、ラベルの個数をカウントし、総数で割る
def create_standard_softlabel(image_names, data):
    soft_label = {}
    for i in range(len(image_names)):
        image_name = image_names[i]
        label_count = [0 for _ in range(5)]
        for j in range(len(label)):
            if data[j] == image_name:
                label_count[label[j]] += 1
        
        count_sum = sum(label_count)
        for k in range(len(label_count)):
            label_count[k] /= count_sum
        soft_label[image_name] = label_count
    return soft_label

# 総数で割らず、softmax関数に入力する
def create_softmax_softlabel(image_names, data):
    soft_label = {}
    count = []
    for i in range(len(image_names)):
        image_name = image_names[i]
        label_count = [0 for _ in range(5)]
        for j in range(len(label)):
            if data[j] == image_name:
                label_count[label[j]] += 1
        count.append(label_count)

    softmax = nn.Softmax(dim=1)
    softmax_count = softmax(torch.tensor(count, dtype=torch.float32)).tolist()
    for i in range(len(image_names)):
        soft_label[image_names[i]] = softmax_count[i]
    return soft_label

soft_label = create_softmax_softlabel(image_names, data)
    
# GradCam
robot_names = ['man', 'woman', 'cyborg_man', 'cyborg_woman', 'cyborg', 'robot_with_al_1', 'robot_with_al_2', 'robot_without_al_1', 'robot_without_al_2', 'thing']
robot_display_names = ['man', 'woman', 'cyborg_man', 'cyborg_woman', 'cyborg', 'w_1', 'w_2', 'wo_1', 'wo_2', 'sphere']
# 上のコードではtrain_test_split内でconvert('RGB')が行われていたので、この場合は自分で行う必要がある
test_image = list(map(lambda x: Image.open('./robot_image/old/' + x + '.png').convert('RGB'), robot_names))
test_data = list(map(transform['test'], test_image))
test_data = torch.stack(test_data, dim=0)
net.eval()
fig = plt.figure(figsize=(20, 15))
with torch.no_grad():
    images = test_data.to(device)
    y_pred_prob = net(images).detach()
    softmax = nn.Softmax(dim=1)
    y_pred_prob = softmax(y_pred_prob).tolist()
    for i in range(10):
        ax = fig.add_subplot(5, 6, 3*i+1)
        ax.axis('off')
        ax.imshow(test_image[i])
        ax2 = fig.add_subplot(5, 6, 3*i+2)
        ax2.bar([0, 1, 2, 3, 4], y_pred_prob[i])
        ax2.set_xticks([0, 1, 2, 3, 4]) 
        ax2.set_xticklabels(['original', '1.15', '1.25', '1.35', '2'])
        ax2.set_ylim(0, 1)
        ax3 = fig.add_subplot(5, 6, 3*i+3)
        ax3.bar([0, 1, 2, 3, 4], soft_label[robot_names[i] + '.png'])
        ax3.set_xticks([0, 1, 2, 3, 4]) 
        ax3.set_xticklabels(['original', '1.15', '1.25', '1.35', '2'])
        ax3.set_ylim(0, 1)
        if i == 0 or i == 1:
            ax.set_title("Image", fontsize=18)
            ax.text(0.35, -0.1, robot_display_names[i], fontsize=15, transform=ax.transAxes)
            ax2.set_title("Model output", fontsize=18)
            ax3.set_title("Target label", fontsize=18)
        else:
            ax.set_title(robot_display_names[i], fontsize=15, y=-0.15)
plt.savefig('./figure/SP_inference.png')