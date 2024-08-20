import cv2
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import mediapipe as mp

################################
wCam, hCam = 640, 480
################################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 400
volPer = 0
colorVol = (255, 0, 0)

# Initialize Mediapipe Hand tracking
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=1,
                      min_detection_confidence=0.7,
                      min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()

    # Convert image to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the image with Mediapipe
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLandmarks in results.multi_hand_landmarks:
            # Get the index finger and thumb tip coordinates
            thumbTip = (int(handLandmarks.landmark[mpHands.HandLandmark.THUMB_TIP].x * wCam),
                        int(handLandmarks.landmark[mpHands.HandLandmark.THUMB_TIP].y * hCam))
            indexTip = (int(handLandmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].x * wCam),
                        int(handLandmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].y * hCam))

            # Calculate the distance between the thumb and index finger
            length = np.linalg.norm(np.array(thumbTip) - np.array(indexTip))

            # Map the distance range to the volume range
            vol = np.interp(length, [50, 200], [minVol, maxVol])
            volume.SetMasterVolumeLevel(vol, None)

            # Calculate the volume percentage
            volBar = np.interp(length, [50, 200], [400, 150])
            volPer = np.interp(length, [50, 200], [0, 100])
            volPer = int(volPer)

            # Update the volume bar color
            colorVol = (0, 255, 0) if volPer > 70 else (255, 0, 0)

            # Draw a line and circle on the hand landmarks
            mpDraw.draw_landmarks(img, handLandmarks, mpHands.HAND_CONNECTIONS,
                                  mpDraw.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=4),
                                  mpDraw.DrawingSpec(color=(255, 0, 255), thickness=2))

    # Draw volume bar on image
    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
    cv2.putText(img, f'{volPer}%', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

    # Display the image
    cv2.imshow("Img", img)

    # Check for the 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()
