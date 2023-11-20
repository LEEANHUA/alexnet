from PIL import Image
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
import matplotlib.pyplot as plt
from torchvision import models

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
net.load_state_dict(torch.load('model_weight.pth'))

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
    ])
}

# 別の画像を使って評価
test_robot_names = ['man', 'woman', 'cyborg_man', 'cyborg_woman', 'cyborg', 'robot_with_al_1', 'robot_with_al_2', 'robot_without_al_1', 'robot_without_al_2', 'thing']
# 上のコードではtrain_test_split内でconvert('RGB')が行われていたので、この場合は自分で行う必要がある
test_image_another = list(map(lambda x: Image.open('./robot_image/old/' + x + '.png').convert('RGB'), test_robot_names))
test_data_another = list(map(transform['test'], test_image_another))
test_data_another = torch.stack(test_data_another, dim=0)
net.eval()
with torch.no_grad():
    images = test_data_another.to(device)
    y_pred_prob = net(images)
    y_pred_prob = y_pred_prob.detach().cpu().numpy().tolist()
    fig = plt.figure(figsize=(15, 15))
    for i in range(10):
        ax = fig.add_subplot(5, 4, 2*i+1)
        ax.axis('off')
        ax.imshow(test_image_another[i])
        ax2 = fig.add_subplot(5, 4, 2*i+2)
        ax2.bar([0, 1, 2, 3, 4], y_pred_prob[i])
    plt.savefig('./figure/AlexNet_test.png')