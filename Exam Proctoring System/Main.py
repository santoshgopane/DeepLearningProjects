from numpy.lib.function_base import append
from ReportGenerator import Generate_report
import cv2
import numpy as np
from FaceNet_Login_Verification import User_Login
from mtcnn.mtcnn import MTCNN
from HeadTracker import *
from datetime import datetime
from ReportGenerator import Generate_report
import os
import shutil

def MainFunctionTrigger(username):

    cap = cv2.VideoCapture(1)

    _, frame = cap.read()
    old_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    count = penalty = Applied_penalty = disp_count = disp_msg_count = fluctuation = i = still_vid_cnt = 0
    snap = 0
    captured_ss = False
    mtcnn = MTCNN()

    _, frame1 = cap.read()
    _, frame2 = cap.read()

    while True:

        time1 = datetime.now()
        time2 = time1.strftime('%m-%d-%Y_%H-%M')

        _, image = cap.read()

        try:

            """
            Face Verification Function will be triggered every 10 Sec [Demo purpose]

            IMPORTANT:  Once the user will login, the username is important thing,
            Login page will ask: Username and live captured image -> if passed image output is equal to the username then allow access!

            Once Login is done, Allow to start monitoring. And in Face Verification use the same username,
            if the username passed and face verification output username is same then okay, else take ScreenShot and store.
            
            IF USER IS MISSING THEN ALSO TAKE ScreenShot and save!
            """

            # Face-Verification Start
            # if 0 <= disp_count:# % 100 < 10:

            #     result_face_array = mtcnn.detect_faces(image)
            #     User_identity, box,captured_ss= User_Login(image,result_face_array,username,i)
            #     cv2.rectangle(image, (box[0],box[1]), (box[0]+box[2], box[1]+box[3]), (255,0,0), 2)
            #     cv2.putText(image, User_identity, (box[0],box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

            #     i = i + 1
            # else:
            #     i = 0
            #     print('Face Verification complete/havent started!')

            # disp_count = disp_count + 1
            # Face-Verification - End

            """
            We should be having a how many times the fluctuation happened,
            IF person looks out of frame for straight 10 sec, then apply penalty.
            but to chategorize genuiness and fake
            IMP:- APPLY SOME LOGIC to check the fluctuation of moment!
            if the added fluctuation have happened thrice, then only apply penalty

            to implement, you might need to have few veriables be used in the Header.py only.(Maybe!)
            PS> same logic applies for EYEs tracking too.
            """

            # Head Tracker - Start
            # image_array,gray_frame = Load_frame(image)
            # result_face_array = mtcnn.detect_faces(image_array)
            # penalty,fluctuation,disp_msg_count = Penalty_count(result_face_array,gray_frame,old_gray,penalty,fluctuation,disp_msg_count,image)
            # print(f'\nPenalty: {penalty}, Fluctuation: {fluctuation}, Disp Count: {disp_msg_count}')
            # if fluctuation >= 5:
            #     if disp_msg_count < 20 and fluctuation <= 6:
            #         # pass
            #         cv2.putText(image, 'You are Continuously Looking Outside!', (200,450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            #     else:
            #         disp_msg_count = 0
            #         fluctuation = 0
            #         Applied_penalty = Applied_penalty + 1
            # Head Tracker - End

            # Laughter Detection - Start
            # Code to invoke Laughter Detection.
            # Laughter Detection - End
            
            if penalty >= 30:
                penalty = 0
                Applied_penalty = Applied_penalty + 1

            print('****Applied Penalty: ',Applied_penalty)

            cv2.putText(image, 'Applied Penalty :'+str(Applied_penalty),(30,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            # cv2.putText(image, 'Penalty :'+str(penalty),(30,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

            if captured_ss:
                print('\n****Screen Shot has been taken for false user identified!')
                if not os.path.exists("Report Generate/"+username):
                    os.makedirs("Report Generate/"+username)
                print("Report Generate/"+username+"/Different User found at "+time2+".jpg")
                cv2.imwrite("Report Generate/"+username+"/Different User fount at "+time2+".jpg", image)
                Applied_penalty = Applied_penalty + 1

            # Code to detect Motion - Start!
            diff = cv2.absdiff(frame1, frame2)
            gray_frame = cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)
            gray_blur = cv2.GaussianBlur(gray_frame,(25,25),0)
            _, threshold = cv2.threshold(gray_blur,20,255, cv2.THRESH_BINARY)
            dilated = cv2.dilate(threshold, None, iterations = 3)
            contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # cv2.drawContours(image,contours,-1, (0,255,0),2)
            frame1 = frame2
            _, frame2 = cap.read()

            print('Contours: ',len(contours),still_vid_cnt)

            if len(contours) == 0:
                still_vid_cnt = still_vid_cnt + 1
            else:
                still_vid_cnt = 0
            if still_vid_cnt == 100:
                Applied_penalty = Applied_penalty + 1
                print('\n****Screen Shot has been taken for Still user!')
                cv2.putText(image, "You're staying still for longer time!",(30,80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                if not os.path.exists("Report Generate/"+username):
                    os.makedirs("Report Generate/"+username)
                print("Report Generate/"+username+"/User Still for Longer time"+time2+".jpg")
                cv2.imwrite("Report Generate/"+username+"/User Still for Longer time"+time2+".jpg", image)
            # Code to detect Motion - End
            cv2.imshow("Video",image)
            if cv2.waitKey(10) & 0xFF == ord('q'):

                Generate_report(username,Applied_penalty)
                path = "C:/Deep Learning Projects/Exam Proctoring System/Report Generate/"+username
                shutil.rmtree(path)
                break
            
            snap = 0

        except Exception as e:
            print('\n*****You are out of the frame!',str(e))
            cv2.putText(image, 'You are out of the frame!',(30,80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            snap = snap + 1

            if snap == 10:
                print('\n****Screen Shot has been taken!')
                if not os.path.exists("Report Generate/"+username):
                    os.makedirs("Report Generate/"+username)
                print("Report Generate/"+username+"/Out of frame at "+time2+".jpg")
                cv2.imwrite("Report Generate/"+username+"/Out of frame at "+time2+".jpg", image)
                Applied_penalty = Applied_penalty + 1
                fluctuation = 0

            print('Snap: ',snap)
            cv2.putText(image, 'Applied Penalty :'+str(Applied_penalty),(30,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
            # cv2.putText(image, 'Penalty :'+str(penalty),(30,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow("Video",image)
            pass
            if cv2.waitKey(10) & 0xFF == ord('q'):
                Generate_report(username,Applied_penalty)
                if os.path.exists("Report Generate/"+username):
                    path = "C:/Deep Learning Projects/Exam Proctoring System/Report Generate/"+username
                shutil.rmtree(path)
                break


    cv2.destroyAllWindows()
# MainFunctionTrigger("Santosh_Gopane")