# Flask関連
from flask import Flask, render_template, request, redirect, url_for, abort

# Pytorch関連
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torchvision import models

# PIL, datetime
from PIL import Image, ImageOps
from datetime import datetime

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
net.load_state_dict(torch.load('../model_weight.pth'))

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

net.eval()

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "GET":
        return render_template("index.html")
    if request.method == "POST":
        # アップロードされたファイルをいったん保存する
        f = request.files["file"]
        filepath = "./static/" + datetime.now().strftime("%Y%m%d%H%M%S") + ".png"
        f.save(filepath)
        # 画像ファイルを読み込む
        image = Image.open(filepath).convert('RGB')
        image = transform['test'](image).unsqueeze(0).to(device)
        y_pred_prob = net(image)
        y_pred_label = torch.max(y_pred_prob, 1)[1]
        result = y_pred_label[0].item()
        
        return render_template("index.html", filepath=filepath, result=result)

if __name__ == "__main__":
    app.run(port=8000, debug=True)