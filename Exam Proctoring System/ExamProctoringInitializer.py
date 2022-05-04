from tkinter import *
from functools import partial
import cv2
from PIL import Image,ImageTk
import numpy as np
from FaceNet_Login import *
from mtcnn.mtcnn import MTCNN
from Main import *

global valid_user
facenet_username = ''
global image
mtcnn = MTCNN()
image = ""

def Monitoring():
    LoginWindow.destroy()
    # Here write a code to Open New Window or direct try to run the main.py function

    MainFunctionTrigger(facenet_username)

Monitoring = partial(Monitoring)

def validateLogin(user_name):
    global username
    username = user_name.get()

    # Trigger FaceNet login check here
    valid_user = True
    print("username entered :",username)
    print('FaceNet: ',facenet_username)
    if username == facenet_username and facenet_username != "":
        print('User is beem Authenticated!')
        Button(LoginWindow, text='Capture Image' , height=3,width=20).place(x=70,y=300)
        Label(LoginWindow,text="User Has Been Authenticated by the System. Click on Start Monitoring.",bg="green",height="2",font=("Calibri", 13)).place(x=500,y=450)

        Button(LoginWindow, text='Start Monitoring' , height=3,width=20,command=Monitoring).place(x=650,y=500)
    else:
        Label(LoginWindow,text="    Invalid Login try. Please Try Again. Person's identity doesn't match.   ",bg="red",height="2" ,font=("Calibri", 13)).place(x=500,y=450)

    return valid_user



def UserVerification():
    # print('Image: ',image2)
    print('Image from Video Cap: ',image[-1])
    global facenet_username
    if image != []:
        print('Triggering FaceNet Model to Check Face!')
        try:
            face_detected_img = mtcnn.detect_faces(image)
            print(face_detected_img)
            facenet_username = User_Initial_Login(image, face_detected_img)
            print('Face Net Output: ',facenet_username)

            Label(LoginWindow,text='                                                       ',height="2",width="50", font=("Calibri", 13)).place(x=550,y=440)
        except:
            Label(LoginWindow,text='Please Adjust Your Camera / Light in surrounding!', height="2", font=("Calibri", 13)).place(x=550,y=440)
    return

LoginWindow  = Tk()

LoginWindow.title('OEP Login Page')
LoginWindow.geometry('500x250')

Label(LoginWindow,text='Username',width="8", height="2", font=("Calibri", 13)).place(x=40,y=10)
username = StringVar()
user_name = Entry(LoginWindow,textvariable=username).place(x=140,y=25)

UserVerification = partial(UserVerification)

Button(LoginWindow, text='Capture Image' , height=3,width=20,command=UserVerification).place(x=70,y=300)

"""
Put a message on screen -> Invalid login try. Username does not match with person's identity! (in red)
"""

validateLogin = partial(validateLogin,username)

Button(LoginWindow, text='Login', height=3,width=20,command=validateLogin).place(x=650,y=500)

frame = LabelFrame(LoginWindow)
frame.pack()
disp = Label(frame,height=400,width=600)
disp.pack()

cap = cv2.VideoCapture(1)

while True:
    _ , image = cap.read()
    image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image1 = ImageTk.PhotoImage(Image.fromarray(image1))
    disp['image'] = image1

    LoginWindow.update()
# LoginWindow.mainloop()

# https://www.youtube.com/watch?v=ypwOOtU2qus
