import os
import json
from PIL import Image
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import torch.nn.functional as F
from torch import optim
import matplotlib.pyplot as plt
from torchinfo import summary
from sklearn.metrics import confusion_matrix
import seaborn as sns
from torchvision import models
import random


# データセットの読み込み
data = []
label = []

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

# jsonファイルから主観評価実験の結果を読み込む
directory = '/home/miyamoto/public_html/exp202309/parameter_robopitch2/exp-data/logql/'
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

# dataの重複をなくし辞書順に並べることで、画像の名前のリストを生成
image_names = sorted(list(set(data)), key=str.lower)

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

def randint_2(a, b):
    n = random.randint(a, b)
    m = random.randint(a, b)
    while n == m:
        m = random.randint(a, b)
    return n, m

# 推論用のデータセットを作成
inference_data = []
inference_label = []
for i in range(len(image_names)):
    inference_data.append(Image.open('./robot_image/' + image_names[i]))
    inference_label.append(soft_label[image_names[i]])

inference_data = list(map(transform['test'], inference_data))

inference_data = torch.stack(inference_data, dim=0)

# ソフトラベルなのでLongTensorじゃなくていい
inference_label = torch.Tensor(inference_label)

# 画像データとラベルを結合したデータセットを作成
inference_dataset = TensorDataset(inference_data, inference_label)

# ミニバッチサイズを指定したデータローダーを作成
batch_size = 64
inference_batch = torch.utils.data.DataLoader(dataset=inference_dataset,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=2)

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



pred_list = []
true_list = []

for i in range(10):
    net.eval()
    with torch.no_grad():
        for images, labels in inference_batch:
            images = images.to(device)
            labels = labels.to(device)
            
            y_pred_prob = net(images)
            
            y_pred_labels = torch.max(y_pred_prob, 1)[1]
            hard_labels = torch.max(labels, 1)[1]
            
            pred_list += y_pred_labels.detach().cpu().numpy().tolist()
            true_list += hard_labels.detach().cpu().numpy().tolist()
    
# 混同行列
cm = confusion_matrix(true_list, pred_list)
cm_float = cm.astype(np.float64)
for i in range(5):
    sum = np.sum(cm_float[i])
    cm_float[i] /= sum
plt.figure()
sns.heatmap(cm_float, annot=True, cmap='Blues')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.savefig('./figure/inference_cm.png')