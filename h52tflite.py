import tensorflow
import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model, Input

import os
import cv2
import glob
import numpy as np

from model_cp import mobilenet

os.environ["CUDA_VISIBLE_DEVICES"]="0"

if __name__=="__main__":
    
    # tf.compat.v1.disable_eager_execution()
    # tf.compat.v1.enable_eager_execution()

    weight_path = './checkpoints/tmp/cp-0005.hdf5'
    savepath = './tflite_save/tmp_cp_0005.tflite'

    model = mobilenet()
    model.load_weights("./checkpoints/tmp_cp/cp-0005.hdf5")
    print(model.summary())
    # outputs = model(input_arr)

    ### create new models
    inflow  = layers.Input((100, 100, 1))
    inflow_ = (inflow-127.5)/127.5
    output  = model(inflow_)
    print(output)
    new_model = Model(inputs=[inflow],outputs=[output])

    print("----Start----")
    converter = tf.lite.TFLiteConverter.from_keras_model(new_model)
    # converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    open(savepath, "wb").write(tflite_model)
    print("----Success----")

    # model = MobileNetV2(input_shape=(100,100,1),classes=4)
    # model.load_weights(weight_path)
    # print(model)
    # print(model.summary())
    
    # inflow  = layers.Input((100, 100, 1))
    # inflow_ = (inflow-127.5)/127.5
    # new_model = Model(inputs=[inflow],outputs=[model.])


    # print("----Start----")
    # converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # print("converter", converter)
    # # converter.target_spec.supported_types = [tf.float16]
    # tflite_model = converter.convert()
    # open(savepath, "wb").write(tflite_model)
    # print("----Success----")

    #converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    #converter.inference_input_type = tf.uint8
    #converter.inference_output_type = tf.uint8

    # input_arr = tf.random.uniform((1,100,100,1))
    # model.build(input_shape=(1,100,100,1))
    # model.load_weights(weight_path)
    # outputs = model(input_arr)
