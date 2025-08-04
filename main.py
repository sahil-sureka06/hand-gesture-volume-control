import cv2
import numpy as np
import time
import math
from HandTrackingModule import handDetector

# pycaw for volume control on Windows
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Webcam settings
wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# Initialize hand detector
detector = handDetector(detectionCon=0.8, maxHands=1)

# Initialize pycaw
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None
)
volume = cast(interface, POINTER(IAudioEndpointVolume))
vol_range = volume.GetVolumeRange()
minVol = vol_range[0]
maxVol = vol_range[1]

pTime = 0

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # Thumb tip (id 4)
        x1, y1 = lmList[4][1], lmList[4][2]
        # Index finger tip (id 8)
        x2, y2 = lmList[8][1], lmList[8][2]
        # Center point between fingers
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Draw circles and line
        cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)

        # Calculate distance between fingers
        length = math.hypot(x2 - x1, y2 - y1)

        # Convert length to volume level
        vol = np.interp(length, [30, 200], [minVol, maxVol])
        volBar = np.interp(length, [30, 200], [400, 150])
        volPercent = np.interp(length, [30, 200], [0, 100])

        # Set volume
        volume.SetMasterVolumeLevel(vol, None)

        # Volume bar
        cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 2)
        cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f'{int(volPercent)} %', (40, 450), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        # If fingers very close, highlight center
        if length < 30:
            cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

    # FPS counter
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Hand Volume Control", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
