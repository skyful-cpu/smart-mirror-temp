from threading import Thread
import cv2 as cv
import mediapipe as mp
import numpy as np
import pyautogui

class WebcamStream:
    def __init__(self, src=0, width=1280, height=720):
        # 비디오 캡처를 위한 객체를 만든다.
        self.stream = cv.VideoCapture(src)
        self.stream.set(3, width)
        self.stream.set(4, height)
        self.grabbed, self.frame = self.stream.read()
        
        self.stopped = False

    def start_threading(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # 계속 프레임을 읽어온다
        while True:
            if self.stopped:
                return
                
            self.grabbed, self.frame = self.stream.read()

    def read(self):
        # 가장 최근 읽은 프레임을 반환한다
        return self.frame
    
    def stop(self):
        # 스레드가 종료되었음을 알려준다
        self.stopped = True

def single_function(csv_file_path):
    # 웹캠 번호 (기본값은 0)
    src = 0          

    # 최대 1개의 손을 인식한다                           
    max_num_hands = 1                           

    # 제스처 관련 변수들
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
    file = np.genfromtxt(csv_file_path, delimiter=',')
    angle = file[:,:-1].astype(np.float32)
    label = file[:, -1].astype(np.float32)
    knn = cv.ml.KNearest_create()
    knn.train(angle, cv.ml.ROW_SAMPLE, label)

    # 중복 클릭 방지를 위한 플래그
    is_clicked = False

    # WebcamStream 객체를 생성
    webcam = WebcamStream(src=src, width=400, height=300).start_threading()

    # 멀티스레딩으로 웹캠에서 프레임을 읽어온다
    while True:
        
        frame = webcam.read()
        
        # MediaPipe에 맞게 프레임의 방향, 색을 바꿔준다
        frame = cv.flip(frame, 1)
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        
        result = hands.process(frame)
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        
        if result.multi_hand_landmarks is not None:
            joint = np.zeros((21, 3))
            
            for i, res in enumerate(result.multi_hand_landmarks):
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
            pyautogui.moveTo(71, 676)
            if idx in click_gesture.keys():
                if is_clicked == False:
                    cv.putText(frame, text=click_gesture[idx].upper(), org=(int(res.landmark[0].x * frame.shape[1]), int(res.landmark[0].y * frame.shape[0] + 20)), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                    #pyautogui.click()
                    print("click")
                    is_clicked = True
                    break

            #if idx in scrolldown_gesture.keys():
            #    cv.putText(frame, text=scrolldown_gesture[idx].upper(),
            #                org=(int(res.landmark[0].x * frame.shape[1]), int(res.landmark[0].y * frame.shape[0] + 20)),
            #                fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
            #    pyautogui.scroll(-100)
            #    print("scroll down")

            #if idx in scrollup_gesture.keys():
            #    cv.putText(frame, text=scrollup_gesture[idx].upper(),
            #                org=(int(res.landmark[0].x * frame.shape[1]), int(res.landmark[0].y * frame.shape[0] + 20)),
            #                fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
            #    pyautogui.scroll(100)
            #    #print("scroll up ")
                
            mp_drawing.draw_landmarks(frame, res, mp_hands.HAND_CONNECTIONS)
            pointer_x = res.landmark[0].x
            pointer_y = res.landmark[0].y
            #print(f"{pointer_x} , {pointer_y}")
            
            

            #pyautogui.moveTo(640*pointer_x, 480*pointer_y)
        else:
            is_clicked = False
        
        # 웹캠 촬영을 종료한다
        cv.imshow('webcam stream', frame)
        if cv.waitKey(1) == ord('q'):
            break
        
    webcam.stop()
    cv.destroyAllWindows()
    return {"gesture": "ok"}
