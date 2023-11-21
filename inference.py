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
    ]),
    'display': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
}
    
# GradCam
robot_names = ['man', 'woman', 'cyborg_man', 'cyborg_woman', 'cyborg', 'robot_with_al_1', 'robot_with_al_2', 'robot_without_al_1', 'robot_without_al_2', 'thing']
# 学習データと同じ画像を用いた画像群から１枚ずつランダムに抜き出して、GradCamを用いて分析
test_image = list(map(lambda x: Image.open('./robot_image/' + x + '/' + x + '_' + str(random.randint(1, 10)) + '.png').convert('RGB'), robot_names))
test_data = list(map(transform['test'], test_image))
test_data = torch.stack(test_data, dim=0)
net.eval()
fig = plt.figure(figsize=(20, 15))
with torch.no_grad():
    images = test_data.to(device)
    y_pred_prob = net(images)
    y_pred_prob = y_pred_prob.detach().cpu().numpy().tolist()
    for i in range(10):
        ax = fig.add_subplot(5, 6, 3*i+1)
        ax.axis('off')
        ax.imshow(test_image[i])
        ax2 = fig.add_subplot(5, 6, 3*i+3)
        ax2.bar([0, 1, 2, 3, 4], y_pred_prob[i])
# 購買計算によるパラメータの更新をONにしないと動かない
for param in net.parameters():
    param.requires_grad = True
# show_cam_on_imageに[0, 1]で入力する必要があるため、正則化を行わないtransformを追加して使用
display_data = list(map(transform['display'], test_image))
target_layers = [net.features]
cam = GradCAM(model=net, target_layers=target_layers, use_cuda=torch.cuda.is_available())
grayscale_cams = cam(input_tensor=test_data)
for i in range(10):
    ax3 = fig.add_subplot(5, 6, 3*i+2)
    ax3.axis('off')
    grayscale_cam = grayscale_cams[i]
    visualization = show_cam_on_image(display_data[i].permute(1, 2, 0).numpy(), grayscale_cam, use_rgb=True)
    ax3.imshow(visualization)
plt.savefig('./figure/inference1.png')

# 別の画像を使って評価
# 上のコードではtrain_test_split内でconvert('RGB')が行われていたので、この場合は自分で行う必要がある
test_image = list(map(lambda x: Image.open('./robot_image/old/' + x + '.png').convert('RGB'), robot_names))
test_data = list(map(transform['test'], test_image))
test_data = torch.stack(test_data, dim=0)
net.eval()
fig = plt.figure(figsize=(20, 15))
with torch.no_grad():
    images = test_data.to(device)
    y_pred_prob = net(images)
    y_pred_prob = y_pred_prob.detach().cpu().numpy().tolist()
    for i in range(10):
        ax = fig.add_subplot(5, 6, 3*i+1)
        ax.axis('off')
        ax.imshow(test_image[i])
        ax2 = fig.add_subplot(5, 6, 3*i+3)
        ax2.bar([0, 1, 2, 3, 4], y_pred_prob[i])
for param in net.parameters():
    param.requires_grad = True
display_data = list(map(transform['display'], test_image))
target_layers = [net.features]
cam = GradCAM(model=net, target_layers=target_layers, use_cuda=torch.cuda.is_available())
grayscale_cams = cam(input_tensor=test_data)
for i in range(10):
    ax3 = fig.add_subplot(5, 6, 3*i+2)
    ax3.axis('off')
    grayscale_cam = grayscale_cams[i]
    visualization = show_cam_on_image(display_data[i].permute(1, 2, 0).numpy(), grayscale_cam, use_rgb=True)
    ax3.imshow(visualization)
plt.savefig('./figure/inference2.png')