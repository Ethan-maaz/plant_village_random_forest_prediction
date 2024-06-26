# -*- coding: utf-8 -*-
"""plant village random forest prediction

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1894UNvdaMOqz-PB8DSkjIsc8oeb-Oamm
"""

!unzip drive/MyDrive/plant-village-dataset/plant_disease.zip

pip install split-folders[full]

import splitfolders

input_file="plantvillage dataset/color"
output_file="plantvillage dataset splitted"

splitfolders.ratio(input_file, output=output_file, seed=42, ratio=(.7, .2, .1), group_prefix=None)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

len(os.listdir("/content/plantvillage dataset splitted/train"))

from keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions

train_datagen = ImageDataGenerator(zoom_range=0.5, shear_range=0.3, horizontal_flip=True)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train=train_datagen.flow_from_directory(directory="/content/plantvillage dataset splitted/train",
                                        target_size=(256,256),
                                        batch_size=32,
                                        class_mode='categorical')
val=val_datagen.flow_from_directory(directory="/content/plantvillage dataset splitted/val",
                                        target_size=(256,256),
                                        batch_size=32,
                                        class_mode='categorical')

num_classes = len(train.class_indices)
print(f'Number of classes: {num_classes}')

t_img, label=train.next()

t_img.shape

def plotimage(img_arr, label):
  for im, l in zip(img_arr, label):
    plt.figure(figsize=(5,5))
    img_array = img_to_array(im)
    img_array /= 255.0
    plt.imshow(img_array)
    plt.show()

plotimage(t_img[:3],label[:3])

from keras.layers import Dense, Flatten
from keras.models import Model
from keras.applications.vgg19 import VGG19
import keras

base_model=VGG19(input_shape=(256,256,3), include_top=False)

for layer in base_model.layers:
  layer.trainable=False

base_model.summary()

#compiling the model
x = Flatten()(base_model.output)

x = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=x)
model.compile(optimizer="adam", loss=keras.losses.categorical_crossentropy, metrics=["accuracy"])

from keras.callbacks import ModelCheckpoint, EarlyStopping

es=EarlyStopping(monitor="val_accuracy", min_delta=0.01, patience=3, verbose=1)
mc=ModelCheckpoint(filepath="best_model.h5", monitor="val_accuracy",patience=3,min_delta=0.01 , verbose=1, save_best_only=True)

cb=[es,mc]

his = model.fit_generator(train, steps_per_epoch=16,
                          epochs=50,
                          verbose=1,
                          callbacks=cb,
                          validation_data=val,
                          validation_steps=16)

h=his.history
h.keys()

plt.plot(h['accuracy'])
plt.plot(h['val_accuracy'],c="red")
plt.title("accuracy vs val_accuracy")
plt.show()

plt.plot(h['loss'])
plt.plot(h['val_loss'],c="red")
plt.title("loss vs val_loss")
plt.show()

from keras.models import load_model
model=load_model("best_model.h5")

acc=model.evaluate_generator(val)[1]
print(f"The accuracy of the model is:{acc*100}%")

ref = dict(zip(list(train.class_indices.values()), list(train.class_indices.keys())))

def prediction(path):
    img = load_img(path,target_size=(256,256))
    i=img_to_array(img)
    im=preprocess_input(i)
    img=np.expand_dims(im,axis=0)
    pred=np.argmax(model.predict(img))
    print(ref[pred])

path = "/content/plantvillage dataset splitted/test/Corn_(maize)___Common_rust_/RS_Rust 1576.JPG"
prediction(path)

path = "/content/plantvillage dataset splitted/test/Tomato___Bacterial_spot/10f0b483-25a2-4c13-9054-754e9fe08d18___GCREC_Bact.Sp 3812.JPG"
prediction(path)

train.class_indices