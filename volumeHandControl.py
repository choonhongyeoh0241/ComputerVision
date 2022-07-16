import cv2
import time
import math
import numpy as np
import HandTrackingModule as module
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

wCam, hCam = 1920, 1080

detector = module.handDetector(detectionCon=0.7)

capture = cv2.VideoCapture(0)
capture.set(3, wCam)
capture.set(4, hCam)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volumeRange = volume.GetVolumeRange()

minVolume = volumeRange[0]
maxVolume = volumeRange[1]
vol = 0
volumeBar = 400
volumePercentage = 0

previousTime = 0
currentTime = 0

while True:
    success, img = capture.read()
    img = detector.findHands(img)
    landmarkList = detector.findPosition(img, draw=False)
    if len(landmarkList) != 0:
        print(landmarkList[4], landmarkList[8])

        x1, y1 = landmarkList[4][1], landmarkList[4][2]
        x2, y2 = landmarkList[8][1], landmarkList[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 5)

        length = math.hypot(x2 - x1, y2 - y1)

        vol = np.interp(length, [50, 250], [minVolume, maxVolume])
        volumeBar = np.interp(length, [50, 200], [400, 150])
        volumePercentage = np.interp(length, [50, 200], [0, 100])
        volume.SetMasterVolumeLevel(vol, None)

        if length < 50:
            cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)

    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    cv2.rectangle(img, (50, int(volumeBar)), (85, 400), (0, 255, 0), cv2.FILLED)

    currentTime = time.time()
    fps = 1/(currentTime - previousTime)
    previousTime = currentTime

    cv2.putText(img, f'VOLUME:{int(volumePercentage)}%', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 250, 0), 3)

    cv2.imshow("Cam", img)
    cv2.waitKey(1)