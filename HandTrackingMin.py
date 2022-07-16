import cv2
import mediapipe as mp
import time

capture = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(False)
mpMarks = mp.solutions.drawing_utils

previousTime = 0
currentTime = 0

while True:
    success, img = capture.read()
    imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(imageRGB)
    print(result.multi_hand_landmarks)

    if result.multi_hand_landmarks:
        for handLandMarks in result.multi_hand_landmarks:
            for index_value, landmark in enumerate(handLandMarks.landmark):
                print(index_value, landmark)
                height, word, channel = img.shape
                channelX, channelY = int(landmark.x*word), int(landmark.y*height)
                print(index_value, channelX, channelY)

                if index_value == 0:
                    cv2.circle(img, (channelX, channelY), 20, (255, 0, 255), cv2.FILLED)
                elif index_value == 8:
                    cv2.circle(img, (channelX, channelY), 20, (255, 0, 255), cv2.FILLED)

            mpMarks.draw_landmarks(img, handLandMarks, mpHands.HAND_CONNECTIONS)

    currentTime = time.time()
    fps = 1/(currentTime-previousTime)

    previousTime = currentTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 3)

    cv2.imshow("Cam", img)
    cv2.waitKey(1)












