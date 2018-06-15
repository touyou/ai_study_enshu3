#!/usr/bin/python
import numpy as np
import chainer
from chainer import Function, Variable
from chainer import Link, Chain
import chainer.functions as F
import chainer.links as L

from xml.etree.ElementTree import *
import os
from PIL import Image
from PIL import ImageOps
import math
import sys
import argparse

import random
import cv2
from matplotlib import pylab as plt

imgsize = 100
landmark = 15

def out_size(in_size, ksize, poolsize):
    return math.ceil((in_size - ksize + 1)  / poolsize)

class MyChain(Chain):
    def __init__(self):
        super(MyChain, self).__init__(
            l1=L.Convolution2D(in_channels = 1, out_channels = 16, ksize = 4),
            l2=L.Convolution2D(in_channels = 16, out_channels = 32, ksize = 5),
            l3=L.Convolution2D(in_channels = 32, out_channels = 64, ksize = 5),
            l4=L.Linear((out_size(out_size(imgsize, 4, 2), 5, 2) - 5 + 1)**2*64, 400),
            l5=L.Linear(400, landmark*2)
        )
        
    def __call__(self, x):
        h1 = F.max_pooling_2d(F.relu(self.l1(x)), 2)
        h2 = F.max_pooling_2d(F.relu(self.l2(h1)), 2)
        h3 = F.relu(self.l3(h2))
        h3_reshape = F.reshape(h3, (len(h3.data), int(h3.data.size / len(h3.data))))
        h4 = F.relu(self.l4(h3_reshape))
        return self.l5(h4)

def get_distance(a, b):
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    return math.sqrt(dx * dx + dy * dy)

def get_center(a, b):
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    return [a[0] + dx / 2, a[1] + dy / 2]

def get_box_from_center_and_length(c, l):
    return (c[0] - l, c[1] - l, c[0] + l, c[1] + l)

def read_image_data(xmlfile, data):
    # 読み込み
    tree = parse(xmlfile)
    root = tree.getroot()
    images = root.find("images")

    for image in list(images):
        img = ImageOps.invert(Image.open(image.get("file")).convert('L'))
        for box in list(image):
            parts = {}
            for part in list(box):
                parts[part.get("name")] = [int(part.get("x")), int(part.get("y")), 1]

            notfound = False
            for label in ("C01", "C02", "C03", "L01", "L02", "L03", "L04", "R01", "R02", "R03", "R04", "M01", "M02", "M03", "M04"):
                if label not in parts:
                    print("not exist {} in box:{}, file:{}".format(label, ibox+1, image.get("file")))
                    notfound = True
            if notfound:
                continue

            c = get_center(parts["L01"], parts["R02"])
            l = get_distance(c, parts["C02"]) * 1.7
        
            # 画像切り取りグレイスケールの値の範囲を0から1にする
            cropped = np.array(img.crop(get_box_from_center_and_length(c, l)), dtype=np.float32) / 256.0

            # ランドマークの座標変換
            for part in parts.values():
                # 左上を0,0にする
                part[0] -= c[0] - l # x
                part[1] -= c[1] - l # y

            data.append({'img': cropped, 'parts' : parts})

def data_augmentation(data):
    img = data['img']
    parts = data['parts']

    width = img.shape[1]
    height = img.shape[0]

    center = (width / 2, height / 2)

    dx = parts["R02"][0] - parts["L01"][0]
    dy = - parts["R02"][1] + parts["L01"][1]
    angle0 = math.degrees(math.atan(dy / dx))

    # 2/3の範囲を100*100にする
    scale0 = 100.0 / (width * 2.0 / 3.0)

    # ランダムに変形を加える
    angle = random.uniform(-45, 45) - angle0
    scale = scale0 * random.uniform(0.9, 1.1)

    # 変形後の原点
    #x0 = width / 3.0 * scale

    # アフィン変換
    #  回転、拡大の後に、回転の中心が(50, 50)になるように平行移動
    #  平行移動にランダムな値を加える
    matrix = cv2.getRotationMatrix2D(center, angle, scale) + np.array([[0, 0, -center[0] + 50 + random.uniform(-3, 3)], [0, 0, -center[1] + 50 + random.uniform(-3, 3)]])
    dst = cv2.warpAffine(img, matrix, (100, 100))

    # ランドマーク座標をnumpyの配列に変換
    parts_np = np.array([
        parts["C01"], parts["C02"], parts["C03"],
        parts["L01"], parts["L02"], parts["L03"], parts["L04"],
        parts["R01"], parts["R02"], parts["R03"], parts["R04"],
        parts["M01"], parts["M02"], parts["M03"], parts["M04"]
        ], dtype=np.float32)

    # ランドマークの座標変換
    parts_converted = parts_np.dot(matrix.T) / 100.0

    # ランダムに反転
    if random.randint(0, 1) == 1:
        dst = cv2.flip(dst, 1)
        for i in range(len(parts_converted)):
            parts_converted[i][0] = 1.0 - parts_converted[i][0]

        parts_converted = np.array([
            parts_converted[2], parts_converted[1], parts_converted[0], # C
            parts_converted[8], parts_converted[7], parts_converted[9], parts_converted[10], # R -> L
            parts_converted[4], parts_converted[3], parts_converted[5], parts_converted[6], # L -> R
            parts_converted[12], parts_converted[11], parts_converted[13], parts_converted[14] # M
            ], dtype=np.float32)

    # 変換されたデータを返す
    return {'img' : dst, 'parts' : parts_converted}


def show_img_and_landmark(img, parts):
    plt.imshow(1.0 - img, cmap='gray')
    for t in parts[0:3]:
        plt.plot(t[0]*100, t[1]*100, 'or')
    for t in parts[3:7]:
        plt.plot(t[0]*100, t[1]*100, 'og')
    for t in parts[7:11]:
        plt.plot(t[0]*100, t[1]*100, 'ob')
    for t in parts[11:15]:
        plt.plot(t[0]*100, t[1]*100, 'oy')
    plt.axis([0, 100, 100, 0])
    plt.show()

#!/usr/bin/python
import numpy as np
import chainer
from chainer import cuda, Function, Variable
from chainer import optimizers, serializers
from chainer import Link, Chain
import chainer.functions as F
import chainer.links as L

from xml.etree.ElementTree import *
import os
from PIL import Image
from PIL import ImageOps
import math
import sys
import argparse
from datetime import datetime

import random
import cv2
from matplotlib import pylab as plt

from facial_landmark import *

parser = argparse.ArgumentParser(description='Train facial landmark')
parser.add_argument('xmlfile', type=str, help='xmlfile created by imglab tool')
parser.add_argument('testfile', type=str,
                    help='xmlfile created by imglab tool')
parser.add_argument('--batchsize', '-b', type=int, default=16,
                    help='Number of images in each mini-batch')
parser.add_argument('--iteration', '-i', type=int,
                    default=100, help='Number of iteration times')
parser.add_argument('--epoch', '-e', type=int, default=1,
                    help='Number of epoch times')
parser.add_argument('--initmodel', '-m', default='',
                    help='Initialize the model from given file')
parser.add_argument('--resume', '-r', default='',
                    help='Resume the optimization from snapshot')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
args = parser.parse_args()

model = MyChain()

model.to_gpu()

#optimizer = optimizers.SGD(lr=args.lr)
optimizer = optimizers.RMSprop(lr=args.lr)
#optimizer = optimizers.Adam()
optimizer.setup(model)
# optimizer.add_hook(chainer.optimizer.WeightDecay(0.0005))

# Init/Resume
if args.initmodel:
    print('Load model from', args.initmodel)
    serializers.load_npz(args.initmodel, model)
if args.resume:
    print('Load optimizer state from', args.resume)
    serializers.load_npz(args.resume, optimizer)

# 訓練データ読み込み
train_data = []
read_image_data(args.xmlfile, train_data)

# テストデータ読み込み
test_data = []
read_image_data(args.testfile, test_data)

#import matplotlib.pyplot as plt
#data = train_data[0]
#img = data[0]
#parts = data[1]
# print(parts)
#plt.imshow(img, 'gray')
# plt.show()

# for i in range(len(train_data)):
#    data = data_augmentation(train_data[i])
#
#    print(i)
#    print(data['parts'])


def mini_batch_data(train_data):
    img_data = []
    t_data = []
    for j in range(args.batchsize):
        data = data_augmentation(train_data[random.randrange(len(train_data))])
        img_data.append(data['img'])
        t_data.append(data['parts'])
        # for debug
        #show_img_and_landmark(data['img'], data['parts'])

    x = Variable(cuda.to_gpu(np.array(img_data, dtype=np.float32)))
    x = F.reshape(x, (args.batchsize, 1, imgsize, imgsize))

    t = Variable(cuda.to_gpu(np.array(t_data, dtype=np.float32)))
    t = F.reshape(t, (args.batchsize, landmark*2))

    return x, t


def save_model(sufix=""):
    # Save the model and the optimizer
    print('save the model')
    serializers.save_npz('model' + sufix, model)
    print('save the optimizer')
    serializers.save_npz('state' + sufix, optimizer)


print("{}, start training".format(datetime.now()))

for epo in range(args.epoch):
    sum_loss = 0

    for i in range(args.iteration):
        # ミニバッチ入力データ
        x, t = mini_batch_data(train_data)

        # 順伝播
        y = model(x)

        # 誤差逆伝播
        model.cleargrads()
        loss = F.mean_squared_error(y, t)
        loss.backward()
        optimizer.update()

        # loss
        sum_loss += float(loss.data)

        # for debug
        # print(loss.data)
        # print(y.data)
        # print(t.data)

    # 検証
    # ミニバッチ入力データ
    x, t = mini_batch_data(test_data)
    # 順伝播
    y = model(x)
    loss = F.mean_squared_error(y, t)

    print("{}, epoch = {}, train loss = {}, test loss = {}".format(datetime.now(), epo+1, sum_loss / args.iteration, loss.data), flush=True)

    # Save the model and the optimizer
    if (epo+1) % 10000 == 0 or epo == args.epoch - 1:
        save_model(str(epo+1))

    # for debug
    #show_img_and_landmark(cuda.to_cpu(x.data)[0][0], cuda.to_cpu(y.data)[0].reshape((landmark,2)))
