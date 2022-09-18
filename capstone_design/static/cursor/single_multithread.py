# 멀티 스레딩을 위한 라이브러리
from __future__ import print_function
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import argparse
import imutils

# FPS 표시를 위한 라이브러리
import time

# PyAutoGUI, OpenCV, MediaPipe
import cv2
import mediapipe as mp
import numpy as np
import pyautogui

pyautogui.FAILSAFE = False
max_num_hands = 1
gesture = {
    0:'fist', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five',
    6:'six', 7:'rock', 8:'spiderman', 9:'yeah', 10:'ok',
}
click_gesture = {0:'click' }
scrolldown_gesture = {5:'scroll down'}
scrollup_gesture = {9:'scroll up'}

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Gesture recognition model
file = np.genfromtxt('gesture_train.csv', delimiter=',')
angle = file[:,:-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)

# created a *threaded* video stream, allow the camera sensor to warmup,
# and start the FPS counter
print('[INFO] sampling THREADED frames from webcam...')
vs = WebcamVideoStream(src=0).start()

#cap = cv2.VideoCapture(0)
#cap.set(3,400) #     
#cap.set(4,300)

prevTime = 0

# fps 정보 저장을 위한 list
framerates = []

while True:

    img = vs.read()
    img = imutils.resize(img, width=400)

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hands.process(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    ###########################
    # get current time (sec)
    curTime = time.time()

    # curTime - prevTime ?
    # = returnTime
    sec = curTime - prevTime
    # update prevTime with curTime
    prevTime = curTime

    # frame = 1 / sec
    # 1 / time per frame
    fps = 1/(sec)

    # string that contains fps
    str = "%0.1f" % fps
    framerates.append(str)

    # show
    cv2.putText(img, str, (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
    ###################################

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 3))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z]

            # Compute angles between joints
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
            v = v2 - v1 # [20,3]
            # Normalize v
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # Get angle using arcos of dot product
            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

            angle = np.degrees(angle) # Convert radian to degree

            # Inference gesture
            data = np.array([angle], dtype=np.float32)
            ret, results, neighbours, dist = knn.findNearest(data, 3)
            idx = int(results[0][0])

        # Draw gesture result
        if idx in click_gesture.keys():
            cv2.putText(img, text=click_gesture[idx].upper(), org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
            pyautogui.click()
            print("click")

        if idx in scrolldown_gesture.keys():
            cv2.putText(img, text=scrolldown_gesture[idx].upper(),
                        org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
            pyautogui.scroll(-100)
            print("scroll down")

        if idx in scrollup_gesture.keys():
            cv2.putText(img, text=scrollup_gesture[idx].upper(),
                        org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
            pyautogui.scroll(100)
            print("scroll up ")




        mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)
        pointer_x = res.landmark[0].x
        pointer_y = res.landmark[0].y
        #print(f"{pointer_x} , {pointer_y}")
        
        

        pyautogui.moveTo(640*pointer_x, 480*pointer_y)



    cv2.imshow('Game', img)
    if cv2.waitKey(1) == ord('q'):
        break

vs.release()
cv2.destroyAllWindows()

f = open('framerates.csv', 'w')

for idx in range(len(framerates)):
    f.write(framerates[idx] + '\n')
   
f.close()