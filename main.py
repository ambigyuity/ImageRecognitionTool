# Kruskal's algorithm in Python

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
import matplotlib.pyplot as plt

#TF_VERSION= tf.version.VERSION
#print(TF_VERSION)

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch, logs={}):
        if(logs.get('accuracy')> 0.95):
            print("\n Reached 95% accuracy! Discontinue training!")
            self.model.stop_training=True

def cameraInput(model):
    cv2.namedWindow("SAMSAN TECH")
    vc = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    print(vc)

    if vc.isOpened():  # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    print(rval)
    while rval:
        # frame is the image
        cv2.imshow("SAMSAN TECH", frame)
        #TODO: MODEL.PREDICT(IMAGE) predict image shown
        labelText= 'Prediction'
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame,
                    labelText,
                    (50, 50),
                    font, 1,
                    (0, 255, 255),
                    2,
                    cv2.LINE_4)
        cv2.imshow("SAMSAN TECH", frame)
        rval, frame = vc.read()
        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            break

    vc.release()
    cv2.destroyWindow("preview")
    model.summary()

def main():
    callbacks= myCallback()
    cifar10= keras.datasets.cifar10
    (train_images, train_labels), (test_images,test_labels)= cifar10.load_data()

    train_images= train_images.reshape(50000,32,32,3)
    test_images= test_images.reshape(10000,32,32,3)
    train_images, test_images= train_images/255.0, test_images/255.0

    model = keras.Sequential(
        [
            #keras.layers.Conv2d(FILTERS, FILTER DIMENSION(3,3), ACTIVATION, INPUTSHAPE)
            #keras.layers.MaxPooling2D(POOLSIZEX, POOLSIZEY)
            keras.layers.Conv2D(64,(3,3), activation='relu', input_shape=(32,32,3)),
            keras.layers.MaxPooling2D(2,2),
            keras.layers.Flatten(),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ]
    )

    model.summary()

    model.compile(optimizer=tf.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])


    #data_generator= ImageDataGenerator()


    history= model.fit(train_images, train_labels, epochs=20, callbacks=[callbacks])

    model.evaluate(test_images, test_labels)

    cameraInput(model)


if __name__== "__main__":
    main()



