import warnings
import os

warnings.simplefilter("ignore", category=FutureWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import cv2
from os import listdir
import matplotlib.pyplot as plt

from PIL import Image
from mtcnn.mtcnn import MTCNN

from keras.models import load_model
from numpy import savez_compressed, expand_dims, load

from sklearn.svm import SVC
from sklearn.preprocessing import Normalizer

# from keras_vggface.utils import preprocess_input
# from keras_vggface.vggface import VGGFace
# from scipy.spatial.distance import cosine

import pickle

mtcnn = MTCNN()

KerasModel = load_model(
    "C:/Computer Vision Projects/FaceDetection/Model/facenet_keras.h5"
)


def DetectFace(Filename, Size=((160, 160))):

    UserImage = cv2.imread(Filename)
    UserImage = cv2.cvtColor(UserImage, cv2.COLOR_BGR2RGB)
    ImageArray = np.array(UserImage)
    FaceDetails = mtcnn.detect_faces(ImageArray)
    try:
        x, y, width, height = FaceDetails[0]["box"]
        x1, y1, x2, y2 = abs(x), abs(y), x + width, y + height
        FacePoints = ImageArray[y1:y2, x1:x2]
        CroppedFace = Image.fromarray(FacePoints)
        CroppedFace = CroppedFace.resize(Size)
        FaceArray = np.array(CroppedFace)
    except:
        FaceArray = []
        print("Could not detect face for:", Filename)
    return FaceArray


def LoadDataset(Directory):
    x, y = [], []
    for subdir in listdir(Directory):

        path = Directory + subdir + "/"

        AllFacesArray = []
        for File in listdir(path):
            FaceArray = DetectFace(path + File)
            if FaceArray != []:
                AllFacesArray.append(FaceArray)

        x.extend(AllFacesArray)
        Labels = [subdir for _ in range(len(AllFacesArray))]
        y.extend(Labels)
        # print(Labels)

    return np.array(x), np.array(y)


def GetFaceEmbeddings(Model, FacePixels):
    FacePixels = FacePixels.astype("float32")
    Mean, Std = FacePixels.mean(), FacePixels.std()
    FacePixels = (FacePixels - Mean) / Std
    Sample = expand_dims(FacePixels, axis=0)
    # print("Sample:", Sample)
    Yhat = Model.predict(Sample)
    return Yhat[0]


TrainData = LoadDataset("D:/Deep Learning/Face Verification Training Data/")
x_train, y_train = TrainData
# x_test, y_test = TestingData
new_train_x = []
for x_pixels in x_train:
    # print("here")
    embedding = GetFaceEmbeddings(KerasModel, x_pixels)
    new_train_x.append(embedding)
new_train_x = np.array(new_train_x)

XTrain, YTrain = new_train_x, y_train

InEncoder = Normalizer(norm="l2")
XTrain = InEncoder.transform(XTrain)

Model = SVC(kernel="linear", probability=True, gamma="auto", random_state=15)
Model.fit(XTrain, YTrain)

pickle.dump(
    Model,
    open(
        "C:/Deep Learning Projects/Face Verification Application/model/FaceDetectionModelV2.model",
        "wb",
    ),
)
print("Model has been saved Successfully!")
