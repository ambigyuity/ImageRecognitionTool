#
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt


# TF_VERSION= tf.version.VERSION
# print(TF_VERSION)

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.95):
            print("\n Reached 95% accuracy! Discontinue training!")
            self.model.stop_training = True


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
        # TODO: MODEL.PREDICT(IMAGE) predict image shown

        im = Image.fromarray(frame, 'RGB')
        im = im.resize((32, 32))
        img_array = np.array(im)
        img_array = np.expand_dims(img_array, axis=0)

        # Change labelText to predictionArray or something
        labelText = model.predict(img_array)[0]
        predictiontext = ""

        # Verify class assignments. Currently it is set alphabetically
        if labelText[0] == 1:
            predictiontext = "airplane"
        elif labelText[1] == 1:
            predictiontext = "automobile"
        elif labelText[2] == 1:
            predictiontext = "bird"
        elif labelText[3] == 1:
            predictiontext = "cat"
        elif labelText[4] == 1:
            predictiontext = "deer"
        elif labelText[5] == 1:
            predictiontext = "dog"
        elif labelText[6] == 1:
            predictiontext = "frog"
        elif labelText[7] == 1:
            predictiontext = "horse"
        elif labelText[8] == 1:
            predictiontext = "ship"
        elif labelText[9] == 1:
            predictiontext = "truck"
        else:
            predictiontext = None

        # labelText= 'Prediction'
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame,
                    predictiontext,
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
    callbacks = myCallback()
    cifar10 = keras.datasets.cifar10
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    train_images = train_images.reshape(50000, 32, 32, 3)
    test_images = test_images.reshape(10000, 32, 32, 3)
    train_images, test_images = train_images / 255.0, test_images / 255.0

    model = keras.Sequential(
        [
            # keras.layers.Conv2d(FILTERS, FILTER DIMENSION(3,3), ACTIVATION, INPUTSHAPE)
            # keras.layers.MaxPooling2D(POOLSIZEX, POOLSIZEY)
            keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Dropout(0.2),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(96, (3, 3), activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(128, (3, 3), activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.BatchNormalization(),
            keras.layers.Flatten(),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ]
    )

    model.summary()

    model.compile(optimizer=tf.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # data_generator= ImageDataGenerator()
    # TODO: IMPORT IMAGES FROM ZIP AND ALL THAT JAZZ

    # TODO: DATA AUGMENTATION
    history = model.fit(train_images, train_labels, epochs=20, callbacks=[callbacks])

    model.evaluate(test_images, test_labels)

    cameraInput(model)


if __name__ == "__main__":
    main()

