import tensorflow as tf

import os
import cv2
import random
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFilter

from model import mobilenet
from model_res import mobilenet_res
from model_vgg import mobilenet_vgg

random.seed(2021)

os.environ["CUDA_VISIBLE_DEVICES"]="4"

def get_list_from_filenames(file_path):
    with open(file_path,'r',) as f:
        lines = [one_line.strip('\n') for one_line in f.readlines()]
    return lines

if __name__=="__main__":

    checkpoint_path = "./checkpoints/tmp/cp-{epoch:04d}.hdf5"
    LOG_DIR = "./checkpoints/tmp_fitlogs/"

    batch_size = 256
    num_epochs = 5
    initial_learning_rate = 0.005

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

    # model = mobilenet_vgg(channels=[12,12,24,24,36,36,48,48,128])
    # model = mobilenet_res(channels=[12,12,24,24,36,36,48,48,128])
    model = mobilenet(channels=[12,12,12,12,24,24,24,24,36,36,36,36,48,48,48,128])

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