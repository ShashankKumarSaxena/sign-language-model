import numpy as np
import pandas as pd
import tensorflow as tf
import os, cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import gc

from keras import layers, callbacks, utils, applications, optimizers, Sequential

path = "dataset"
files = os.listdir(path)

files.sort()
print(files)

image_array = []
label_array = []

for i in tqdm(range(len(files))):
    sub_file = os.listdir(path + "/" + files[i])
    
    for j in range(len(sub_file)):
        file_path = path + "/" + files[i] + "/" + sub_file[j]
        image = cv2.imread(file_path)
        image = cv2.resize(image, (96, 96))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_array.append(image)

        label_array.append(i)

image_array = np.array(image_array)
label_array = np.array(label_array, dtype="float")

# Splitting dataset into test and train
X_train, X_test, Y_train, Y_test = train_test_split(image_array, label_array, test_size=0.2, random_state=42)

# Freeing up memory
del image_array, label_array
gc.collect()

model = Sequential()

pretrained_model = tf.keras.applications.EfficientNetB0(input_shape=(96, 96, 3), include_top=False)
model.add(pretrained_model)

model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dropout(0.3))
model.add(layers.Dense(1))

model.build(input_shape=(None, 96, 96, 3))

model.summary()

model.compile(optimizer="adam", loss="mae", metrics=["mae"])
ckp_path = "trained_model/model.weights.h5"

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=ckp_path,
    monitor="val_mae",
    mode="auto",
    save_best_only=True,
    save_weights_only=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    factor=0.9,
    monitor="val_mae",
    model="auto",
    cooldown=0,
    patience=5,
    verbose=1,
    min_lr=1e-6
)

Epochs = 100
Batch_size = 32

history = model.fit(
    X_train,
    Y_train,
    validation_data=(X_test, Y_test),
    batch_size=Batch_size,
    epochs=Epochs,
    callbacks=[model_checkpoint, reduce_lr]
)

model.load_weights(ckp_path)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_model)

prediction_val = model.predict(X_test, batch_size=32)
print(prediction_val[:10])
print(Y_test[:10])
