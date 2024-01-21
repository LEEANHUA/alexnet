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

# 0-4のハードなラベルをソフトラベルに置き換える
new_data = []
new_label = []
for i in range(len(image_names)):
    new_label.append(soft_label[data[i]])
    new_data.append(Image.open('./robot_image/' + data[i]))
        
# 学習データとテストデータを分割
train_data, test_data, train_label, test_label = train_test_split(new_data, new_label, test_size=0.2)
train_data = list(map(transform['train'], train_data))
test_data = list(map(transform['test'], test_data))
        
# train_data配列に[3, 縦, 横]のTensorが1600個入っているので、これを[1600, 3, 縦, 横]のTensorに変換
train_data = torch.stack(train_data, dim=0)
test_data = torch.stack(test_data, dim=0) # test_dataも同様

# ソフトラベルなのでLongTensorじゃなくていい
train_label = torch.Tensor(train_label)
test_label = torch.Tensor(test_label)

# 画像データとラベルを結合したデータセットを作成
train_dataset = TensorDataset(train_data, train_label)
test_dataset = TensorDataset(test_data, test_label)

# ミニバッチサイズを指定したデータローダーを作成
batch_size = 64
train_batch = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=2)
test_batch = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=2)

# ニューラルネットワークの定義
num_classes = 5 # 0~4で分類

class AlexNet(nn.Module):
    def __init__(self, num_classes= num_classes):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True), # inplace=Trueにすると、追加の出力を割り当てず、入力を書き換える
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ネットワークのロード
# CPUとGPUのどちらを使うかを指定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# net = AlexNet().to(device)
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

summary(model=net, input_size=(batch_size, 3, 224, 224))

# 損失関数の定義
criterion = nn.CrossEntropyLoss()

# 最適化関数の定義
optimizer = optim.Adam(net.parameters())

# 損失と正解率を保存するリストを作成
train_loss_list = []
train_accuracy_list = []
test_loss_list = []
test_accuracy_list = []
pred_list = []
true_list = []

#学習の実行
epoch = 30
for i in range(epoch):
    print('--------------------------------------------')
    print("Epoch: {}/{}".format(i+1, epoch))
    
    train_loss = 0
    train_accuracy = 0
    test_loss = 0
    test_accuracy = 0
    
    # -----------------学習パート-----------------
    net.train()
    for images, labels in train_batch:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        y_pred_prob = net(images)
        loss = criterion(y_pred_prob, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        y_pred_labels = torch.max(y_pred_prob, 1)[1]
        hard_labels = torch.max(labels, 1)[1]
        train_accuracy += torch.sum(y_pred_labels == hard_labels).item() / len(hard_labels)
    
    epoch_train_loss = train_loss / len(train_batch)
    epoch_train_accuracy = train_accuracy / len(train_batch)
    
    # -----------------評価パート-----------------
    net.eval()
    with torch.no_grad():
        for images, labels in test_batch:
            images = images.to(device)
            labels = labels.to(device)
            
            y_pred_prob = net(images)
            loss = criterion(y_pred_prob, labels)
            
            test_loss += loss.item()
            
            y_pred_labels = torch.max(y_pred_prob, 1)[1]
            hard_labels = torch.max(labels, 1)[1]
            test_accuracy += torch.sum(y_pred_labels == hard_labels).item() / len(hard_labels)
            
            pred_list += y_pred_labels.detach().cpu().numpy().tolist()
            true_list += hard_labels.detach().cpu().numpy().tolist()
    
    epoch_test_loss = test_loss / len(test_batch)
    epoch_test_accuracy = test_accuracy / len(test_batch)
    
    print("Train_Loss: {:.4f} Train_Accuracy: {:.4f}".format(
        epoch_train_loss, epoch_train_accuracy))
    print("Test_Loss: {:.4f} Test_Accuracy: {:.4f}".format(
        epoch_test_loss, epoch_test_accuracy))
    
    train_loss_list.append(epoch_train_loss)
    train_accuracy_list.append(epoch_train_accuracy)
    test_loss_list.append(epoch_test_loss)
    test_accuracy_list.append(epoch_test_accuracy)
    
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
plt.savefig('./figure/AlexNet_confusionmatrix.png')

# 損失  
plt.figure()
plt.title('Train and Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(range(1, epoch+1), train_loss_list, color='blue',
         linestyle='-', label='Train_Loss')
plt.plot(range(1, epoch+1), test_loss_list, color='red',
         linestyle='--', label='Test_Loss')
plt.legend()
plt.savefig('./figure/AlexNet_loss.png')

# 正解率
plt.figure()
plt.title('Train and Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(range(1, epoch+1), train_accuracy_list, color='blue',
         linestyle='-', label='Train_Accuracy')
plt.plot(range(1, epoch+1), test_accuracy_list, color='red',
         linestyle='--', label='Test_Accuracy')
plt.legend()
plt.savefig('./figure/AlexNet_accuracy.png')

# 重みとバイアスのみ保存
torch.save(net.state_dict(), 'model_weight.pth')