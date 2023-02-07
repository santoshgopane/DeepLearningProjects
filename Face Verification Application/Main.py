import os
import warnings

warnings.simplefilter("ignore", category=FutureWarning)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import numpy as np
from numpy import expand_dims
from PIL import Image
from sklearn.preprocessing import Normalizer

# import tensorflow as tf
from tensorflow import keras
import pickle

# from keras.models import load_model

FaceTracker = keras.models.load_model(
    "C:/Deep Learning Projects/Face Verification Application/Model/NNmodelv3.h5"
)

KerasModel = keras.models.load_model(
    "C:/Computer Vision Projects/FaceDetection/Model/facenet_keras.h5"
)

LoadedModel = pickle.load(
    open(
        "C:/Deep Learning Projects/Face Verification Application/model/FaceDetectionV6.model",
        "rb",
    )
)

def CaptureFaceEmbedding(Model, FacePixels):
    FacePixels = FacePixels.astype("float32")
    Mean, Std = FacePixels.mean(), FacePixels.std()
    FacePixels = (FacePixels - Mean) / Std
    Sample = expand_dims(FacePixels, axis=0)
    Yhat = Model.predict(Sample)
    return Yhat


cap = cv2.VideoCapture(0)

while cap.isOpened():

    _, frame = cap.read()
    print(frame.shape)
    # frame = frame
    frame = frame[50:500, 50:500, :]
    print(frame.size)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (120, 120))
    # resized = tf.image.resize(rgb, (120, 120))

    yhat = FaceTracker.predict(np.expand_dims(resized / 255, 0))
    print("Prediction:", yhat[0])
    sample_coords = yhat[1][0]
    print("first:", tuple(np.multiply(sample_coords[:2], [450, 450]).astype(int)))
    print("second:", tuple(np.multiply(sample_coords[2:], [450, 450]).astype(int)))
    x1, y1 = tuple(np.multiply(sample_coords[:2], [450, 450]).astype(int))
    x2, y2 = tuple(np.multiply(sample_coords[2:], [450, 450]).astype(int))
    print(x2 - x1)
    print(y2 - y1)
    if x2 - x1 < 100 or y2 - y1 < 100:
        print("INVALID")
    face = np.array(frame)
    cropped_face = Image.fromarray(face)
    cropped_face_resize = cropped_face.resize(((160, 160)))
    face_array = np.array(cropped_face_resize)

    embedding = CaptureFaceEmbedding(KerasModel, face_array)
    in_encoder = Normalizer(norm="l2")
    new_embedding = in_encoder.transform(embedding.reshape(-1, 1))
    embedding_result = LoadedModel.predict(new_embedding.reshape(1, 128))
    print("Prediction Result:", embedding_result)

    if yhat[0] > 0.9:
        # Controls the main rectangle
        cv2.rectangle(
            frame,
            tuple(np.multiply(sample_coords[:2], [450, 450]).astype(int)),
            tuple(np.multiply(sample_coords[2:], [450, 450]).astype(int)),
            (255, 0, 0),
            1,
        )
        cv2.rectangle(
            frame,
            tuple(
                np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int), [0, -30])
            ),
            tuple(
                np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int), [80, 0])
            ),
            (255, 0, 0),
            -1,
        )
        cv2.putText(
            frame,
            embedding_result[0],
            tuple(
                np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int), [0, -5])
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.imshow("FaceTracker", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
