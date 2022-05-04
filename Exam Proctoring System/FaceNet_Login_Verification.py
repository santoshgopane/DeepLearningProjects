import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import Normalizer
import cv2
from PIL import Image
from numpy import expand_dims
from keras.models import load_model
import pickle
import datetime

old_value = ""

def get_face_embedding(model,face_pixels):
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()# Pixel value standardization
    face_pixels = ( face_pixels - mean ) / std
    sample = expand_dims(face_pixels,axis = 0)
    yhat = model.predict(sample)
    return yhat[0]

model = load_model('Model/Facenet_keras.h5')
with open('Finalized_FaceVerificationV0.1.model','rb') as file:
    svc_model = pickle.load(file)

def User_Login(img,result_face_array,old_value,i): #img
    # result_face_array = mtcnn.detect_faces(img)
    # print(result_face_array)
    time1 = datetime.datetime.now()

    x, y, width, height = result_face_array[0]['box']
    x1, y1, x2, y2 = abs(x), abs(y), x + width, y + height
    image_arr = np.array(img)
    face = image_arr[y1:y2, x1:x2]
    cropped_face = Image.fromarray(face)
    cropped_face_resize = cropped_face.resize(((160,160)))
    face_array = np.array(cropped_face_resize)
    
    embedding = get_face_embedding(model,face_array)
    in_encoder = Normalizer(norm ='l2')
    new_embedding = in_encoder.transform(embedding.reshape(-1,1))
    embedding_result = svc_model.predict(new_embedding.reshape(1,128))

    print('Embedding Result: ',embedding_result)
    new_value = embedding_result[0]
    print('Old Value: ',old_value)
    print('New Value: ',new_value)
    capture_it = False
    if old_value != new_value:
        capture_it = True
    # time1 = datetime.datetime.now()
    # time = time1.strftime('%m-%d-%Y_%H-%M-%S')

    # if old_value != new_value and old_value != "":
    #     cv2.putText(img, 'Different person detected, Adjust the camera. Penalty will be applied in 10 sec for same occurance!',(20,530), cv2.FONT_HERSHEY_PLAIN, 1.1, (0, 0, 255), 2, cv2.LINE_AA)
    #     penalty = penalty + 1
    #     if penalty >= 20:
    #         penalty = 0
    #         Applied_penalty = Applied_penalty + 2
    #     print('I in function: ',i)
    #     if i == 1:
    
    #         cv2.imwrite("Report Generate/Different person detected instead of "+old_value+" at "+time+".jpg", img) #timestamp!
    #         print('Photo has been taken')
    #     print('Different person detected, Adjust the camera. Penalty will be applied in 10 sec for same occurance!')
    # elif old_value or old_value == "":
    #     cv2.putText(img, 'Face Verification in progress...',(550,530), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    #     pass
    bounding_box = result_face_array[0]['box']
    return new_value, bounding_box,capture_it #embedding_result[0]

    # except Exception as e:
        # cv2.putText(img, "You're out of Frame",(550,530), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        # cv2.imshow("Video",img)
        # time2 = time1.strftime('%m-%d-%Y_%H-%M')
        # penalty = penalty + 1
        # if penalty >= 20:
        #     penalty = 0
        #     Applied_penalty = Applied_penalty + 2
        # print('I in function: ',i)
        # if i == 1:
        #     cv2.imwrite("Report Generate/"+old_value+" is out of frame at "+time2+".jpg", img)
        # print("You're out of Frame! with error: "+e)
        # return "", [],Applied_penalty,penalty