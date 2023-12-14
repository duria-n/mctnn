import cv2
import numpy as np
from keras.layers import Conv2D, Input, MaxPool2D, Flatten, Dense, Permute
from keras.layers.advanced_activations import PReLU
from keras.models import Model

import config
import utils


def create_PNet(weight_path):
    input = Input(shape=[None,None,3])

    x = Conv2D(10,(3,3),strides=1,padding='valid',name='conv1')(input)
    #PRelu相比普通的Relu函数有可学习参数，其中shared_axis指定了共享参数的轴
    x = PReLU(shared_axes=[1,2],name='PRelu1')(x)
    #这里MaxPool2D对应的默认strides=2
    x = MaxPool2D(pool_size=2)(x)

    x = Conv2D(16,[3,3],strides=1,padding='valid',name='conv2')(x)
    x = PReLU(shared_axes=[1,2],name='PRelu2')(x)

    x = Conv2D(32,(3,3),strides=1,padding='valid',name='conv3')(x)
    x = PReLU(shared_axes=[1,2],name='PRelu3')(x)

    classifier = Conv2D(2,(1,1),activation='softmax',name='conv4-1')(x)

    bbox_regression = Conv2D(4,(1,1),name='conv4-2')(x)
    #两种写法都可以，看自己的爱好
    model = Model(input,[classifier,bbox_regression],name='Pnet')
    model.load_weights(weight_path,by_name=True)
    return model

def create_RNet(weight_path):
    input = Input(shape=[24,24,3])

    x = Conv2D(28,(3,3),strides=1,padding='valid',name='conv1')(input)
    x = PReLU(shared_axes=[1,2],name='PRelu1')(x)
    x = MaxPool2D(pool_size=3,strides=2,padding='same')(x)

    x = Conv2D(48,(3,3),strides=1,padding='valid',name='conv2')(x)
    x = PReLU(shared_axes=[1,2],name='PRelu2')(x)
    x = MaxPool2D(pool_size=3,strides=2)(x)

    x = Conv2D(64,(2,2),strides=1,padding='valid',name='conv3')(x)
    x = PReLU(shared_axes=[1,2],name='PRelu3')(x)

    x = Permute((3, 2, 1))(x)

    x = Flatten()(x)
    x = Dense(128,name='conv4')(x)
    x = PReLU(name='PRelu4')(x)

    classifier = Dense(2,activation='softmax',name='conv5-1')(x)
    bbox_regression = Dense(4,name='conv5-2')(x)

    model = Model([input],[classifier,bbox_regression],name='Rnet')
    model.load_weights(weight_path,by_name=True)
    return model

def create_ONet(weight_path):
    input = Input(shape=[48,48,3])

    x = Conv2D(32,(3,3),strides=1,padding='valid',name='conv1')(input)
    x = PReLU(shared_axes=[1,2],name='PRelu1')(x)
    x = MaxPool2D(pool_size=3,strides=2,padding='same')(x)

    x = Conv2D(64,(3,3),strides=1,padding='valid',name='conv2')(x)
    x = PReLU(shared_axes=[1,2],name='PRelu2')(x)
    x = MaxPool2D(pool_size=3,strides=2)(x)

    x = Conv2D(64,(3,3),strides=1,padding='valid',name='conv3')(x)
    x = PReLU(shared_axes=[1,2],name='PRelu3')(x)
    x = MaxPool2D(pool_size=2)(x)

    x = Conv2D(128,(2,2),strides=1,padding='valid',name='conv4')(x)
    x = PReLU(shared_axes=[1,2],name='PRelu4')(x)
    #这里将x原来的第一维度数据放到了第三维度
    x = Permute((3,2,1))(x)

    x = Flatten()(x)
    x = Dense(256,name='conv5')(x)
    x = PReLU(name='PRelu5')(x)


    classifier = Dense(2,activation='softmax',name='conv6-1')(x)
    bbox_regression = Dense(4,name='conv6-2')(x)
    landmark_regression = Dense(10,name='conv6-3')(x)

    model = Model([input],[classifier,bbox_regression,landmark_regression],name='Onet')
    model.load_weights(weight_path,by_name=True)
    return model


class mtcnn():
    def __init__(self):
        self.pNet = create_PNet(config.pNet_path)
        self.oNet = create_ONet(config.oNet_path)
        self.rNet = create_RNet(config.rNet_path)

    def summary(self):
        print('Pnet.summary:')
        self.pNet.summary()
        print('Rnet.summary:')
        self.rNet.summary()
        print('Onet.summary:')
        self.oNet.summary()
    def detect(self,image,threshold):
        """
                这里是进行推理，因此进行尺度计算，将多尺度信息穿入Pnet进行推理，以增加推理的准确度
        :param image:
        :param threshold:
        :return:
        """
        copy_image = (image.copy()-127.5) / 127.5
        origin_h, origin_w, _ = copy_image.shape

        scales = utils.calculateScales(image)

        outs = []
        #在不同尺度下，将图像调整到对应尺度
        for scale in scales:
            hs, ws = int(origin_h*scale), int(origin_w*scale)
            scale_image = cv2.resize(copy_image,(ws,hs))
            inputs = scale_image.reshape(1,*scale_image.shape)
            #Pnet处理数据
            output = self.pNet.predict(inputs)
            outs.append(output)

        image_num = len(scales)
        rectangles = []
        #在不同尺度下，将图像输入到Pnet进行处理，输出的是所有尺度下的Pnet结果
        for i in range(image_num):
            #outs[i]表示outs中的第i个维度的输出，outs[i][0]该维度下的0号输出，outs[i][0][0]表示存在人脸的概率
            cls_probs = outs[i][0][0][:,:,1]
            #这里1表示bbox的信息
            roi = outs[i][1][0]

            outh,outw = cls_probs.shape
            out_side = max(outh,outw)
            #输出储存人脸分布概率图的形状
            print(cls_probs.shape)
            #处理疏忽数据
            rectangle = utils.detect_face_12net(cls_probs,roi,out_side,1/scales[i],origin_w,origin_h,threshold[0])
            rectangles.extend(rectangle)
        #这里针对不同尺度下可能会出现冗余框进行优化
        rectangles = utils.NMS(rectangles,0.7)

        if  len(rectangles) == 0:
            return rectangles

        #进入Rnet处理范围,Rnet的输入数据是处理后的原始图像，在copy_image上进行裁切，这里裁切的位置信息来自Pnet的输出
        predict_24_batch = []
        for rectangle in rectangles:
            crop_image = copy_image[int(rectangle[1]):int(rectangle[3]),int(rectangle[0]):int(rectangle[2])]
            scale_image = cv2.resize(crop_image,(24,24))
            predict_24_batch.append(scale_image)

        predict_24_batch = np.array(predict_24_batch)
        out = self.rNet.predict(predict_24_batch)

        cls_probs = out[0]
        cls_probs = np.array(cls_probs)
        roi_prob = out[1]
        roi_prob = np.array(roi_prob)
        rectangles = utils.filter_face_24net(cls_probs,roi_prob,rectangles,origin_w,origin_h,threshold[1])

        if len(rectangles) == 0:
            return rectangles
        #进入Onet处理范围
        predict_batch = []
        for rectangle in rectangles:
            crop_image = copy_image[int(rectangle[1]):int(rectangle[3]),int(rectangle[0]):int(rectangle[2])]
            scale_image = cv2.resize(crop_image,(48,48))
            predict_batch.append(scale_image)

        predict_batch = np.array(predict_batch)
        output = self.oNet.predict(predict_batch)
        cls_probs = output[0]
        roi_prob = output[1]
        pts_prob = output[2]
        rectangles = utils.filter_face_48net(cls_probs,roi_prob,pts_prob,rectangles,origin_w,origin_h,threshold[2])
        return rectangles
