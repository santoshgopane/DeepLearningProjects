import cv2

cap = cv2.VideoCapture(1)

_, frame1 = cap.read()
_, frame2 = cap.read()

while True:

    diff = cv2.absdiff(frame1, frame2)
    gray_frame = cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray_frame,(25,25),0)
    _, threshold = cv2.threshold(gray_blur,20,255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(threshold, None, iterations = 3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )
    cv2.drawContours(frame1,contours,-1, (0,255,0),2)

    cv2.imshow('Video',frame1)
    frame1 = frame2
    _, frame2 = cap.read()
    print('test: ',contours)
    for contour in contours:
        (x, y, w, h)=cv2.boundingRect(contour)

        if cv2.contourArea(contour) < 5000:
            continue
        # cv2.rectangle(frame1, (x, y), (x+w, y+h), (0,255,0), 1)
        cv2.putText(frame1, 'MOVING!',(30,80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # #Calculating the difference and image thresholding
    # delta=cv2.absdiff(baseline_image,gray_frame)
    # # Finding all the contours
    # (contours,_)=cv2.findContours(threshold,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # # Drawing rectangles bounding the contours (whose area is > 5000)