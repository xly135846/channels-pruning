import tensorflow as tf
import keras.backend as K
from kerassurgeon import identify
from kerassurgeon.operations import delete_channels

import os
import cv2
import random
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFilter

from model_vgg import mobilenet_vgg

random.seed(2021)

os.environ["CUDA_VISIBLE_DEVICES"]="4"

def get_list_from_filenames(file_path):
    with open(file_path,'r',) as f:
        lines = [one_line.strip('\n') for one_line in f.readlines()]
    return lines

# layers_name = ["conv_1",             # add
#                "block_1_conv_1",     
#                "block_1_deconv_1",
#                "block_1_conv_2",     # add
#                "block_2_conv_1",
#                "block_2_deconv_1",
#                "block_2_conv_2",
#                "block_3_conv_1",
#                "block_3_deconv_1",
#                "block_3_conv_2",
#                "block_4_conv_1",
#                "block_4_deconv_1",
#                "block_4_conv_2",
#                "block_5_conv_1",
#                "block_5_deconv_1",
#                "block_5_conv_2",
#                "block_6_conv_1",
#                "block_6_deconv_1",
#                "block_6_conv_2",
#                "block_7_conv_1",
#                "block_7_deconv_1",
#                "block_7_conv_2",
#                "block_8_conv_1",
#                "block_8_conv_2",]

layers_name = ['conv_1',
               'conv_2',
               'conv_3',
               'conv_4',
               'conv_5',
               'conv_6',
               'conv_7',
               'conv_8',
               'block_8_conv_1',
               ]

if __name__=="__main__":

    model = mobilenet_vgg(channels=[12,12,24,24,36,36,48,48,128])
    model.load_weights("./checkpoints/tmp_vgg/cp-0005.hdf5")
    print(model.summary())

    batch_size = 256
    num_epochs = 5
    initial_learning_rate = 0.001

    checkpoint_path = "./checkpoints/tmp_vgg_pruning/cp-{epoch:04d}.hdf5"
    LOG_DIR = "./checkpoints/tmp_vgg_pruning_fitlogs/"

    train_image_arr = np.load("./val_list_image_array.npy")
    train_label_arr = np.load("./val_list_one_hot.npy")

    val_image_arr = np.load("./val_list_image_array.npy")
    val_label_arr = np.load("./val_list_one_hot.npy")

    train_dataset = tf.data.Dataset.from_tensor_slices((train_image_arr, train_label_arr))
    val_dataset   = tf.data.Dataset.from_tensor_slices((val_image_arr, val_label_arr))

    train_dataset = train_dataset.shuffle(buffer_size=20000)
    # train_dataset = train_dataset.repeat(2)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    val_dataset   = val_dataset.batch(batch_size)
    val_dataset   = val_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    for i in range(len(layers_name)):
        print('------',layers_name[i],'------')
        layer_conv = model.get_layer(name=layers_name[i])
        apoz_layer_conv = identify.get_apoz(model, layer_conv, train_image_arr)
        high_apoz_channels_conv = identify.high_apoz(apoz_layer_conv, "both")
        # print(apoz_layer_conv,high_apoz_channels_conv)
        model = delete_channels(model, layer_conv, high_apoz_channels_conv)
        print('success ------',layers_name[i],'------')
        # except:
        #     pass
        #     print('failed ------',layers_name[i],'------')

    print(model.summary())

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate,
                                                                decay_steps=15,
                                                                decay_rate=0.95,
                                                                staircase=True
                                                                )

    model.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=initial_learning_rate),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy']
                  )

    checkpointer = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                      save_weights_only=True,
                                                      verbose=1,
                                                      )

    reduce_lr = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=1)

    HISTORY = model.fit(train_dataset, 
                        epochs=num_epochs,
                        validation_data=val_dataset,
                        callbacks=[checkpointer, reduce_lr, tensorboard_callback],
                        shuffle=True,
                        )

    print("success")