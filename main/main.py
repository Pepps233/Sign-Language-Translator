import math
import time
import threading
import pyttsx3

from cvzone.ClassificationModule import Classifier
from cvzone.HandTrackingModule import HandDetector
import cv2
import numpy as np

#
videoCap = cv2.VideoCapture(0)

detector = HandDetector(maxHands=1, detectionCon=0.6, minTrackCon=0.5)
classifier = Classifier("D:/PyCharmProjects/SignLanguageTranslator/Model/keras_model.h5",
                        "D:/PyCharmProjects/SignLanguageTranslator/Model/labels.txt")

offset = 20
imgSize = 300

labels = ["A", "B", "C", "Confirm", "Cancel"]

scanned = False
showGray = False
waiting = True
waitStartTime = time.time()
waitTime = 0.5

engine = pyttsx3.init()
speechString = ""
speechIndex = 0


def toggleScanned():
    global showGray
    global scanned
    showGray = True
    time.sleep(2)
    scanned = False
    showGray = False


def initTTS():
    global engine
    engine.setProperty('rate', 130)
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)


scannedThread = threading.Thread(target=toggleScanned)
initTTS()

while True:
    success, img = videoCap.read()
    successTwo, imgTwo = videoCap.read()

    if showGray:
        cv2.imshow("output", cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        cv2.imshow("outputTwo", imgTwo)
    else:
        cv2.imshow("output", img)
        cv2.imshow("outputTwo", imgTwo)

    if not scanned:
        if not waiting:
            initialHands, img = detector.findHands(img, draw=True, flipType=True)

            if initialHands:
                waiting = True
                waitStartTime = time.time()
        else:
            elapsed = time.time() - waitStartTime
            if elapsed >= waitTime:
                waiting = False
                hands, imgTwo = detector.findHands(imgTwo, draw=True, flipType=True)
                if hands:

                    hand1 = hands[0]  # getting hand detected
                    lmList1 = hand1["lmList"]  # list of 21 landmarks for the first hand
                    x, y, w, h = hand1["bbox"]  # bounding box around the hand (x,y,w,h coordinates)
                    center1 = hand1['center']  # center coordinates of the first hand
                    handType1 = hand1["type"]  # type of hand (left, right)
                    processedImg = np.ones((imgSize, imgSize, 3), np.uint8) * 255

                    try:
                        # x & y represents the top-left coordinates of the bounding box
                        # cropping will start from x - offset (left to x) to offset + the width (right of bounding box)
                        # y - offset (up from y) to offset + height (lower of bounding box)
                        croppedImg = imgTwo[y - offset:y + h + offset, x - offset:x + w + offset]
                        # 0, 0 represents top left
                        # the height and width will be from 0 to the height and width of cropped img
                        if h > w:
                            ratio = imgSize / h
                            calculatedWidth = math.ceil(ratio * w)
                            if calculatedWidth == 300:
                                calculatedWidth -= 1
                            resizedImg = cv2.resize(croppedImg, (calculatedWidth, imgSize))
                            widthOffset = math.ceil((imgSize - calculatedWidth) / 2)
                            processedImg[:, widthOffset: widthOffset + calculatedWidth] = resizedImg
                            prediction, index = classifier.getPrediction(processedImg)
                            speechIndex = int(index)
                            print(labels[index])
                        else:
                            ratio = imgSize / w
                            calculatedHeight = math.ceil(ratio * h)
                            if calculatedHeight == 300:
                                calculatedHeight -= 1
                            resizedImg = cv2.resize(croppedImg, (imgSize, calculatedHeight))
                            heightOffset = math.ceil((imgSize - calculatedHeight) / 2)
                            processedImg[heightOffset: heightOffset + calculatedHeight, :] = resizedImg
                            prediction, index = classifier.getPrediction(processedImg)
                            speechIndex = int(index)
                            print(labels[index])
                        # good data will not exceed the width or height of 300
                        scanned = True
                        waitStartTime = time.time()

                        speechString += labels[speechIndex]
                        if speechIndex == 3:
                            print("speaking")
                            engine.say(speechString.replace("Confirm", ""))
                            engine.runAndWait()
                        if speechIndex == 4:
                            print("clearing")
                            speechString = ""

                        if croppedImg.shape[0] < 300 and croppedImg.shape[1] < 300:
                            cv2.imshow("processing...", processedImg)

                        if not scannedThread.is_alive():
                            scannedThread = threading.Thread(target=toggleScanned)
                            scannedThread.start()

                    except Exception as e:
                        print(e)

    cv2.waitKey(1)
