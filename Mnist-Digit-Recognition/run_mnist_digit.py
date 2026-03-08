# mnist_digit_recognizer.py
import sys
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import os

model_path = "mnist_cnn_model.h5"

if not os.path.exists(model_path):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1,28,28,1)/255.0
    x_test = x_test.reshape(-1,28,28,1)/255.0
    model = Sequential([
        layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64,(3,3),activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(128,activation='relu'),
        layers.Dense(10,activation='softmax')
    ])
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    model.fit(x_train,y_train,epochs=10,batch_size=32,validation_split=0.1)
    model.save(model_path)
else:
    model = load_model(model_path)

def preprocess(image_path):
    img = Image.open(image_path).convert('L')
    img = ImageOps.invert(img)
    img = np.array(img)
    img = np.where(img>128,255,0).astype(np.uint8)
    coords = np.argwhere(img)
    if coords.shape[0]==0:
        img = np.zeros((28,28),dtype=np.uint8)
    else:
        y0,x0 = coords.min(axis=0)
        y1,x1 = coords.max(axis=0)
        img = img[y0:y1+1,x0:x1+1]
    img = Image.fromarray(img).resize((20,20),Image.Resampling.LANCZOS)
    img = np.array(img)
    padded = np.pad(img,((4,4),(4,4)),"constant",constant_values=0)
    padded = padded/255.0
    return padded.reshape(1,28,28,1)

image = preprocess(sys.argv[1])
prediction = model.predict(image)
print(np.argmax(prediction))