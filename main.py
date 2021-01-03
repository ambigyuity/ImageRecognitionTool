import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import ResNet50
import urllib.request
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# weights_url = "https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
# weights_file = "inception_v3.h5"
# urllib.request.urlretrieve(weights_url, weights_file)

def preprocess_image_input(input_images):
  input_images = input_images.astype('float32')
  output_ims = tf.keras.applications.resnet50.preprocess_input(input_images)
  return output_ims

def feature_extractor(inputs):
    feature_extractor = tf.keras.applications.resnet.ResNet50(input_shape=(224, 224, 3),
                                                              include_top=False,
                                                              weights='imagenet')(inputs)
    return feature_extractor

def classifier(inputs):
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024, activation="relu")(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dense(10, activation="softmax", name="classification")(x)
    return x


def final_model(inputs):
    resize = tf.keras.layers.UpSampling2D(size=(7, 7))(inputs)

    resnet_feature_extractor = feature_extractor(resize)
    classification_output = classifier(resnet_feature_extractor)

    return classification_output


def define_compile_model():
    inputs = tf.keras.layers.Input(shape=(32, 32, 3))

    classification_output = final_model(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=classification_output)

    model.compile(optimizer='SGD',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def cameraInput(model,classes):
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
        im = im.resize((32,32))
        img_array = np.array(im)
        img_array = np.expand_dims(img_array, axis=0)

        # Change labelText to predictionArray or something
        labelText = model.predict(img_array)[0]
        Prediction= np.argmax(labelText)
        #print(Prediction)
        predictiontext = ""

        # Verify class assignments. Currently it is set alphabetically

        #print(labelText)
        labelText=Prediction

        predictiontext= classes[labelText]

        # if labelText ==  0:
        #     predictiontext = "airplane"
        # elif labelText == 1:
        #     predictiontext = "automobile"
        # elif labelText==2:
        #     predictiontext = "bird"
        # elif labelText==3:
        #     predictiontext = "cat"
        # elif labelText==4:
        #     predictiontext = "deer"
        # elif labelText==5:
        #     predictiontext = "dog"
        # elif labelText==6:
        #     predictiontext = "frog"
        # elif labelText==7:
        #     predictiontext = "horse"
        # elif labelText==8:
        #     predictiontext = "ship"
        # elif labelText==9:
        #     predictiontext = "truck"
        # else:
        #     predictiontext = None
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



BATCH_SIZE = 32
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

(training_images, training_labels) , (validation_images, validation_labels) = tf.keras.datasets.cifar10.load_data()
train_X = preprocess_image_input(training_images)
valid_X = preprocess_image_input(validation_images)

model = define_compile_model()

model.summary()

EPOCHS=1
history = model.fit(train_X, training_labels, epochs=EPOCHS, validation_data = (valid_X, validation_labels), batch_size=64)

loss, accuracy = model.evaluate(valid_X, validation_labels, batch_size=64)

cameraInput(model, classes)
