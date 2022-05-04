import cv2
import numpy as np
from EyesTracker import Eye_tracker

lk_params = dict(winSize = (20,20),
            maxLevel = 2,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.3))


def Load_frame(image):
    gray_frame = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # img = image.copy()
    img_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image_arr = np.array(img_rgb)
    return image_arr,gray_frame

def Penalty_count(result,gray_frame,old_gray,penalty,fluctuation,disp_msg_count,MainImage):
    # try:
    nose = result[0]['keypoints']['nose']
    old_points = np.array([nose],dtype=np.float32) # Keeping it float32 is IMPORTANT!

    new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, old_points, None, **lk_params)

    old_gray = gray_frame.copy()
    old_points = new_points

    X1,Y1 = new_points.ravel()

    bounding_box = result[0]['box']

    # To increment the penalty or not(at Main file)!
    # if penalty >= 20:
    #     penalty = 0
    #     Applied_penalty = Applied_penalty + 1

    if X1 - bounding_box[0] <= 30: #Actual Left Side!
        penalty = penalty + 1

    elif bounding_box[0]+bounding_box[2] - X1 <= 30:
        penalty = penalty + 1

    elif  X1 - bounding_box[0] >30 and bounding_box[0]+bounding_box[2] - X1 > 30 and penalty != 0: #fluctuation
        print('Non zero penalty')
        penalty_e,fluctuation = Eye_tracker(result,MainImage,penalty,fluctuation)
        # penalty_e = 0
        if penalty_e > penalty:
            penalty = penalty + 1
        else:
            penalty = 0
            fluctuation = fluctuation + 1

    elif X1 - bounding_box[0] > 30 and bounding_box[0]+bounding_box[2] - X1 > 30:
        # penalty = 0
        print('Triggering Eyes Tracker from Head Movement Tracker!')
        penalty,fluctuation = Eye_tracker(result,MainImage,penalty,fluctuation)
        # pass
        # if penalty != 0:
        #     penalty = penalty + 1
    print(f"*****Penalty from Eyes Tracker {penalty}")
        # Need to add Fluctuation to Eyes tracker too, Add similar code to Eyes Tracker!

    if fluctuation >= 5:
        disp_msg_count = disp_msg_count + 1
        # print('Message Count: ',disp_msg_count)

    return penalty, fluctuation, disp_msg_count