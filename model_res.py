import tensorflow
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras import layers

import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"]="4"

def mobilenet_res(channels=[12,12,24,24,36,36,48,48,128]):
    inputs = Input((100, 100, 1))
    conv_1 = Conv2D(channels[0], 
                    (3,3),
                    strides=(2,2),
                    padding='same',
                    activation='relu',
                    name='conv_1')(inputs)
    conv_1_bn = BatchNormalization()(conv_1)
    # conv_1_ac = ReLU(6.)(conv_1_bn)
    # print("111",conv_1_bn.shape)

    conv_2 = Conv2D(channels[1], 
                (1,1),
                strides=(1,1),
                padding='same',
                activation='relu',
                name='conv_2')(conv_1_bn)
    conv_2_bn = BatchNormalization()(conv_2)
    # print("conv_2_bn", conv_2_bn.shape)
    block_2_add = conv_1_bn + conv_2_bn
    # print("555",block_1_add.shape)

    conv_3 = Conv2D(channels[2], 
                (3,3),
                strides=(2,2),
                padding='same',
                activation='relu',
                name='conv_3')(block_2_add)
    conv_3_bn = BatchNormalization()(conv_3)
    # print("conv_3_bn", conv_3_bn.shape)

    conv_4 = Conv2D(channels[3], 
                (1,1),
                strides=(1,1),
                padding='same',
                activation='relu',
                name='conv_4')(conv_3_bn)
    conv_4_bn = BatchNormalization()(conv_4)
    # print("conv_4_bn", conv_4_bn.shape)
    block_4_add = conv_3_bn + conv_4_bn

    conv_5 = Conv2D(channels[4], 
                (3,3),
                strides=(2,2),
                padding='same',
                activation='relu',
                name='conv_5')(block_4_add)
    conv_5_bn = BatchNormalization()(conv_5)
    # print("conv_5_bn", conv_5_bn.shape)

    conv_6 = Conv2D(channels[5], 
                (1,1),
                strides=(1,1),
                padding='same',
                activation='relu',
                name='conv_6')(conv_5_bn)
    conv_6_bn = BatchNormalization()(conv_6)
    # print("conv_6_bn", conv_6_bn.shape)
    block_6_add = conv_5_bn + conv_6_bn

    conv_7 = Conv2D(channels[6], 
                (3,3),
                strides=(2,2),
                padding='same',
                activation='relu',
                name='conv_7')(block_6_add)
    conv_7_bn = BatchNormalization()(conv_7)
    # print("conv_7_bn", conv_7_bn.shape)

    conv_8 = Conv2D(channels[7], 
                (1,1),
                strides=(1,1),
                padding='same',
                activation='relu',
                name='conv_8')(conv_7_bn)
    conv_8_bn = BatchNormalization()(conv_8)
    # print("conv_8_bn", conv_8_bn.shape)
    block_8_add = conv_7_bn + conv_8_bn

    block_8_conv_1 = Conv2D(channels[8], 
                            (1,1),
                            strides=(1,1),
                            padding='same',
                            activation='relu',
                            name='block_8_conv_1')(block_8_add)
    block_8_avg_pool = AveragePooling2D((7,7))(block_8_conv_1)
    block_8_conv_2 = Conv2D(4, 
                            (1,1),
                            strides=(1,1),
                            padding='same',
                            activation='relu',
                            name='block_8_conv_2')(block_8_avg_pool)
    block_8_detect = tf.squeeze(block_8_conv_2, [1, 2])
    block_8_softmax = tf.nn.softmax(block_8_detect, name= 'softmax')
    
    model = Model(inputs=inputs, outputs=block_8_detect)

    return model


if __name__=="__main__":
    
    model = mobilenet_res(channels=[12,12,24,24,36,36,48,48,128])
    model.summary()
    print("success")