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


# データセットの読み込み
data = []
label = []

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
        image = Image.open('./robot_image/' + robot)
        image = image.resize((224, 224)) # 1024*1024から圧縮
        image = torchvision.transforms.functional.to_tensor(image) # PILからTensor[3, 縦, 横]に変換
        data.append(image)
        ans = alist[i]
        label.append(ans)
        
# 学習データとテストデータを分割
train_data, test_data, train_label, test_label = train_test_split(data, label, test_size=0.2)
        
# train_data配列に[3, 縦, 横]のTensorが1600個入っているので、これを[1600, 3, 縦, 横]のTensorに変換
train_data = torch.stack(train_data, dim=0)
test_data = torch.stack(test_data, dim=0) # test_dataも同様
print("Train Data: {}, Test Data: {}".format(train_data.size(), test_data.size()))

# 回帰の場合はlabelもtorch.float32のデータ型に変換 
train_label = torch.Tensor(train_label)
# このままだと[1600], [400]の形でnn.nn.MSELoss()が受け付けないので、[1600, 1], [400, 1]に変換
train_label = torch.reshape(train_label, (-1, 1)) # 引数に-1を入れると、それ以外の引数から自動的に出力配列の値を決める
test_label = torch.Tensor(test_label)
test_label = torch.reshape(test_label, (-1, 1))
print("Train Label: {}, Test Label: {}".format(train_label.size(), test_label.size()))

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
num_classes = 1 # 回帰なので1

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
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

# ネットワークのロード
# CPUとGPUのどちらを使うかを指定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = AlexNet().to(device)

summary(model=net, input_size=(batch_size, 3, 224, 224))

# 損失関数の定義
criterion = nn.MSELoss()
criterion2 = nn.L1Loss()

# 最適化関数の定義
optimizer = optim.Adam(net.parameters())

# 損失と正解率を保存するリストを作成
train_loss_list = []
train_mae_list = []
test_loss_list = []
test_mae_list = []

#学習の実行
epoch = 100
for i in range(epoch):
    print('--------------------------------------------')
    print("Epoch: {}/{}".format(i+1, epoch))
    
    train_loss = 0
    train_mae = 0
    test_loss = 0
    test_mae = 0
    
    # -----------------学習パート-----------------
    net.train()
    for images, labels in train_batch:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        y_pred_prob = net(images)
        loss = criterion(y_pred_prob, labels)
        mae = criterion2(y_pred_prob, labels)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_mae += mae.item()
    
    epoch_train_loss = train_loss / len(train_batch)
    epoch_train_mae = train_mae / len(train_batch)
    
    # -----------------評価パート-----------------
    net.eval()
    with torch.no_grad():
        for images, labels in test_batch:
            images = images.to(device)
            labels = labels.to(device)
            
            y_pred_prob = net(images)
            loss = criterion(y_pred_prob, labels)
            mae = criterion2(y_pred_prob, labels)
            
            test_loss += loss.item()
            test_mae += mae.item()
    
    epoch_test_loss = test_loss / len(test_batch)
    epoch_test_mae = test_mae / len(test_batch)
    
    print("Train_Loss: {:.4f} Train_MAE: {:.4f}".format(
        epoch_train_loss, epoch_train_mae))
    print("Test_Loss: {:.4f} Test_MAE: {:.4f}".format(
        epoch_test_loss, epoch_test_mae))
    
    train_loss_list.append(epoch_train_loss)
    train_mae_list.append(epoch_train_mae)
    test_loss_list.append(epoch_test_loss)
    test_mae_list.append(epoch_test_mae)
    
plt.figure()
plt.title('Train and Test Loas')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(range(1, epoch+1), train_loss_list, color='blue',
         linestyle='-', label='Train_Loss')
plt.plot(range(1, epoch+1), test_loss_list, color='red',
         linestyle='--', label='Test_Loss')
plt.legend()
plt.savefig('./figure/AlexNet_recursion_loss.png')

plt.figure()
plt.title('Train and Test MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.plot(range(1, epoch+1), train_mae_list, color='blue',
         linestyle='-', label='Train_MAE')
plt.plot(range(1, epoch+1), test_mae_list, color='red',
         linestyle='--', label='Test_MAE')
plt.legend()
plt.savefig('./figure/AlexNet_recursion_MAE.png')