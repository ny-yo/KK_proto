#ライブラリの読み込み
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers

#事前に設定するパラメータ
classes = ["cat","dog"]
nb_classes = len(classes)
img_width, img_height = 150, 150

# VGG16のロード。FC層は不要なので include_top=False
input_tensor = Input(shape=(img_width, img_height, 3))
vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

# FC層の作成
top_model = Sequential()
top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(nb_classes, activation='softmax'))

# VGG16とFC層を結合してモデルを作成
vgg_model = Model(vgg16.input, top_model(vgg16.output))

from keras.models import load_model
vgg_model.load_weights('/home/naoki/kk_proto/kk_proto/src_vgg16/Final.h5')

# テスト用のコード
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# 画像を読み込んで予測する
def img_predict(filename):
    # 画像を読み込んで4次元テンソルへ変換
    img = image.load_img(filename, target_size=(img_height, img_width))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    # 学習時にImageDataGeneratorのrescaleで正規化したので同じ処理が必要
    # これを忘れると結果がおかしくなるので注意
    x = x / 255.0   
    #表示
    #plt.imshow(img)
    #plt.show()
    # 指数表記を禁止にする
    np.set_printoptions(suppress=True)

    #画像の人物を予測    
    pred = vgg_model.predict(x)[0]
    #結果を表示する
    #print("file name %s" % filename)
    #print("　　'cat': 0, 'dog': 1")
    #print(pred*100)
    index_max = np.argmax(pred)
    #print(index_max)
    if index_max == 0:
    	label = '猫'
    
    elif index_max == 1:
    	label = '犬'
    
    return label
#import glob
#テスト用の画像が入っているディレクトリのpathを()に入れてください
#test = glob.glob('C:/Users/naosa/kk_proto/test/*')

#for test_img in test:
#    img_predict(test_img)
