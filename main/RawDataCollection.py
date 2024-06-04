import math
import sys
import time

from cvzone.HandTrackingModule import HandDetector
import cv2
import numpy as np
#
videoCap = cv2.VideoCapture(0)

detector = HandDetector(maxHands=1, detectionCon=0.5, minTrackCon=0.3)

offset = 20
imgSize = 300
ct = 0
folder = "D:/PyCharmProjects/SignLanguageTranslator/RawData/Alphabets/D"

while True:
    success, img = videoCap.read()
    hands, img = detector.findHands(img, draw=True, flipType=True)

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
            croppedImg = img[y - offset:y + h + offset, x - offset:x + w + offset]
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
            else:
                ratio = imgSize / w
                calculatedHeight = math.ceil(ratio * h)
                if calculatedHeight == 300:
                    calculatedHeight -= 1
                resizedImg = cv2.resize(croppedImg, (imgSize, calculatedHeight))
                heightOffset = math.ceil((imgSize - calculatedHeight) / 2)
                processedImg[heightOffset: heightOffset + calculatedHeight, :] = resizedImg
            # good data will not exceed the width or height of 300
            if croppedImg.shape[0] < 300 and croppedImg.shape[1] < 300:
                cv2.imshow("test", processedImg)
        except Exception as e:
            print(e)

        cv2.imshow("output", img)

    else:
        cv2.imshow("output", img)

    key = cv2.waitKey(1)
    if key == ord('s'):
        ct += 1
        cv2.imwrite(f'{folder}/image_{time.time()}.jpg', processedImg)
        print(ct)
    if key == ord('q'):
        sys.exit()
