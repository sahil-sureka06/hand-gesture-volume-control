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




# #IMPORTS

# import cv2 
# import time 
# import numpy as np
# import HandTrackingModule as htm
# import math
# # from pycaw.pycaw import AudioUtilities
# import os


# wCam,hCam = 640,480


# cap = cv2.VideoCapture(0)
# cap.set(3,wCam)
# cap.set(4,hCam)
# # i=0
# pTime = 0

# detector = htm.handDetector()

# def set_volume_mac(volume_percent):
#     volume_percent = int(np.clip(volume_percent, 0, 100))  # Clip between 0–100
#     os.system(f"osascript -e 'set volume output volume {volume_percent}'")


# # device = AudioUtilities.GetSpeakers()
# # volume = device.EndpointVolume
# # # print(f"Audio output: {device.FriendlyName}")
# # # volume.GetMute()
# # # volume.GetMasterVolumeLevel()
# # print(volume.GetVolumeRange())
# # # volume.SetMasterVolumeLevel(-20.0, None)

# while(True):
#     success, img = cap.read()

#     img = detector.findHands(img)
#     lmList = detector.findPosition(img,draw=False)
#     if len(lmList) != 0:
#         # print(lmList[4], lmList[8])

#         x1, y1 = lmList[4][1], lmList[4][2]
#         x2, y2 = lmList[8][1], lmList[8][2]
#         cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
#         cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
#         cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
#         cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
#         cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
#         length = math.hypot(x2 - x1, y2 - y1)
#         print(length)



#     cTime = time.time()
#     fps = 1/(cTime-pTime)
#     # print(cTime, pTime, fps)
#     pTime = cTime
#     cv2.putText(img, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

#     cv2.imshow("Video Feed", img)
#     cv2.waitKey(1)
#     # print(i,"hi")
#     # i+=1

# # import mediapipe as mp
# # import math
# # import vlc
# # import os
# # from ctypes import cast, POINTER
# # from comtypes import CLSCTX_ALL
# # from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# # devices = AudioUtilities.GetSpeakers()
# # interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
# # volume = cast(interface, POINTER(IAudioEndpointVolume))
# # vRange = volume.GetVolumeRange()
# # volper = 0  
# # minV = vRange[0]
# # maxV = vRange[1] 
# # media = vlc.MediaPlayer("Nature.mp4")
# # media.play()

# # vol = 0
# # volbar = 350

# # mp_drawing = mp.solutions.drawing_utils
# # mp_hands = mp.solutions.hands


# # with mp_hands.Hands(
# #     min_detection_confidence=0.8,
# #     min_tracking_confidence=0.5) as hands:
# #     pause_flag = True
#     # img = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)
#     # img.flags.writeable = False
#     # results = hands.process(img)

# #         img.flags.writeable = True
# #         img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
# #         cv2.rectangle(img,(380,75),(630,350),(153, 255, 204),1)
# #         x_roi1,y_roi1 = 380,75
# #         x_roi2,y_roi2 = 630,350
        
# #         cv2.rectangle(img,(125,75),(225,350),(153, 255, 204),1)
# #         x_roi3,y_roi3 = 125,75
# #         x_roi4,y_roi4 = 225,350
        
        
# #         if results.multi_hand_landmarks:
# #             for hand_landmarks in results.multi_hand_landmarks:
# #                 mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

# #             x11,y11 = int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x*wCam) ,int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y* hCam) 
# #             x10,y10 = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x*wCam),int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y* hCam)

# #             if (x_roi1<x10) and (x11<x_roi2):
# #                 cv2.putText(img,'Play/Pause controller',(355,50),cv2.FONT_HERSHEY_COMPLEX,0.7,(26, 26, 26),2)
# #                 x3,y3 = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x*wCam) ,int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y* hCam) 
# #                 x4,y4 = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x*wCam),int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y* hCam)

# #                 cv2.line(img,(x3,y3),(x4,y4),(159, 226, 191),3)

# #                 length1 = math.hypot(x3-x4,y3-y4)

# #                 #pause 75-90
# #                 #play > 150

# #                 if length1>200 and pause_flag == False:
# #                     pause_flag = True
# #                     media.play()
# #                     cv2.putText(img,'Play ',(390,150),cv2.FONT_HERSHEY_COMPLEX,0.7,(26, 26, 26),2)
# #                 elif length1<110 and pause_flag == True:
# #                     pause_flag = False
# #                     media.pause()
# #                     cv2.putText(img,'Pause',(390,150),cv2.FONT_HERSHEY_COMPLEX,0.7,(26, 26, 26),2)

# #             elif (x_roi3<x10) and (x11<x_roi4):

# #                 cv2.putText(img,'Volume controller',(125,50),cv2.FONT_HERSHEY_COMPLEX,0.7,(26, 26, 26),2)
# #                 x1,y1 = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x*wCam) ,int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y* hCam) 
# #                 x2,y2 = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x*wCam),int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y* hCam)
# #                 cx = int((x1+x2)/2)
# #                 cy = int((y1+y2)/2)

# #                 cv2.circle(img,(x1,y1),15,(204, 204, 255),cv2.FILLED)
# #                 cv2.circle(img,(x2,y2),15,(204, 204, 255),cv2.FILLED)
# #                 cv2.circle(img,(cx,cy),15,(204, 204, 255),cv2.FILLED)

# #                 cv2.line(img,(x1,y1),(x2,y2),(159, 226, 191),3)

# #                 length = math.hypot(x2-x1,y2-y1)
# #                 #print(length)

# #                 if length<30:
# #                     cv2.circle(img,(cx,cy),15,(0,255,0),cv2.FILLED)

# #                 #hand range = 30-200
# #                 #vol range = -65 - 0

# #                 vol = np.interp(length,[30,200],[minV,maxV])
# #                 volbar = np.interp(length,[30,200],[350,100])
# #                 volper = np.interp(length,[30,200],[0,100])
# #                 #print(vol)
# #                 volume.SetMasterVolumeLevel(vol,None)

# #             cv2.rectangle(img,(50,100),(85,350),(153, 204, 255),3)
# #             cv2.rectangle(img,(50,int(volbar)),(85,350),(102, 255, 255),cv2.FILLED)
# #             cv2.putText(img,f'{int(volper)}%',(40,50),cv2.FONT_HERSHEY_COMPLEX,0.7,(26, 26, 26),2)

# #         cv2.imshow('HAND GESTURE RECOGNITION', img)

# #         if cv2.waitKey(1) & 0xFF == ord('q'):
# #             break

# # cap.release()
# # cv2.destroyAllWindows()