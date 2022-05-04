import cv2

def Eye_tracker(result,resize_img,penalty,fluctuation):
    print('\nHitted Eyes tracker!')
    # try:

    left_eye = result[0]['keypoints']['left_eye']
    right_eye = result[0]['keypoints']['right_eye']

    bounding_box = result[0]['box']
    eye_width = bounding_box[2] / 4
    eye_height = bounding_box[3] * 0.06

    w, h= int(eye_width) - 12, int(eye_height)
    x, y = left_eye
    x1, y1 = x - int(w / 2), y - 6
    left_eye = resize_img[abs(y1):abs(y1+h),abs(x1):abs(x1+w)] #left_top:height,-> [y1:y2,x1:x2]
    # cv2.rectangle(resize_img, (x1, y1),(x1 + w, y1 + h),(255,0,0),1)

    gray_left_eye = cv2.cvtColor(left_eye,cv2.COLOR_BGR2GRAY)
    _, gray_left_threshold = cv2.threshold(gray_left_eye,70,255,cv2.THRESH_BINARY)

    h, w = gray_left_threshold.shape
    left_side_threshold = gray_left_threshold[0:h , 0:int(w/2)]
    left_side_whiteL = cv2.countNonZero(left_side_threshold)

    right_side_threshold = gray_left_threshold[0:h , int(w/2):w]
    right_side_whiteL = cv2.countNonZero(right_side_threshold)

    w, h= int(eye_width) - 12, int(eye_height)
    x, y = right_eye
    x1, y1 = x - int(w / 2), y - 6
    right_eye = resize_img[abs(y1):abs(y1+h),abs(x1):abs(x1+w)] #left_top:height, 
    # cv2.rectangle(resize_img, (x1, y1),(x1 + w, y1 + h),(255,0,0),1)
    gray_right_eye = cv2.cvtColor(right_eye,cv2.COLOR_BGR2GRAY)
    _, gray_right_threshold = cv2.threshold(gray_right_eye,90,255,cv2.THRESH_BINARY)

    h, w = gray_right_threshold.shape
    left_side_threshold = gray_right_threshold[0:h , 0:int(w/2)]
    left_side_whiteR = cv2.countNonZero(left_side_threshold)

    right_side_threshold = gray_right_threshold[0:h , int(w/2):w]
    right_side_whiteR = cv2.countNonZero(right_side_threshold)
    # print('Right side Left eye: ',right_side_whiteL)
    # print('Right side Right eye: ',right_side_whiteR)
    gaze_ratio = 0
    if right_side_whiteL > 0 and right_side_whiteR > 0:
        left_eye_ratio = left_side_whiteL / right_side_whiteL 
        right_eye_ratio = left_side_whiteR / right_side_whiteR 

        gaze_ratio = ( left_eye_ratio + right_eye_ratio ) / 2


    # left_side_look = ( left_side_whiteR + left_side_whiteL ) / 2
    # right_side_look = ( right_side_whiteR + right_side_whiteL ) / 2

    # cv2.putText(resize_img,"Gaze ratio: "+str(gaze_ratio),(50,70),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

    # if penalty >= 20:
    #     penalty = 0
    #     Applied_penalty = Applied_penalty + 1
    # old values left_side_look>=150 & right_side_look>=200 and (right_side_look - left_side_look)>=100
    print("*&*&*Gaze: ",gaze_ratio)
    if gaze_ratio >= 4:
        # cv2.putText(resize_img,"Looking out of frame!",(50,200),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        penalty = penalty + 1
        print('Looking at your Actual Right!',gaze_ratio)

    elif gaze_ratio <= 0.35:
        # cv2.putText(resize_img,"Looking out of frame!",(50,200),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        penalty = penalty + 1
        print('Looking at your Actual Left!',gaze_ratio)

    elif 2 < gaze_ratio > 0.25: #and penalty != 0:
        penalty = 0
        # fluctuation = fluctuation + 1
    # else:
    #     penalty = 0
        # fluctuation and Disp_msg_count

    # else:
        # cv2.putText(resize_img,"Looking at Center",(50,200),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)



    # if left_side_look>=250:
    #     # cv2.putText(resize_img,"Looking left",(50,180),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    #     penalty = penalty + 1
    #     print('left')
    # elif right_side_look>=350 and (right_side_look - left_side_look)>=100:
    #     # cv2.putText(resize_img,"Looking Right",(50,180),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    #     penalty = penalty + 1
    #     print('right')
    # else:
        # cv2.putText(resize_img,"Looking Center",(50,180),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    # print('Penalty Printed before returning penalty in Eyes tracker: ',str(penalty))
    # except Exception as e:
    #     print("***",str(e))
    return penalty,fluctuation