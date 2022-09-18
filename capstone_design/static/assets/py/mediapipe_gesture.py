import cv2 as cv
import numpy as np
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

class mediapipe_gesture:

    def __init__(self):
        self._is_error = False

    def start_gesture(self):
        cap = cv.VideoCapture(0)
        print("start gesture recog")

        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5) as hands:

            while cap.isOpened():
                success, image = cap.read()

                if not success:
                    print("Ignoring empty camera frame.")
                    self._is_error = True
                    break

                image.flags.writeable = False
                image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                results = hands.process(image)

                image.flags.writeable = True
                image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS)
                
                cv.imshow("Mediapipe Hands", cv.flip(image, 1))

                if cv.waitKey(5) & 0xFF == 27:
                    break
        
        cv.destroyAllWindows()
        cap.release()

        if self._is_error:
            self._is_error = False
            return False
        else:
            self._is_error = False
            return True
