import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector


def findDistance(detector, p1, p2, img=None):
    new_p1 = p1[0], p1[1]
    new_p2 = p2[0], p2[1]
    return detector.findDistance(new_p1, new_p2, img)


def isFist(detector, lmList):
    for i in range(8, 21, 4):
        if findDistance(detector, lmList[i], lmList[i-3])[0] > 50:
            return False
    return True


def isBao(detector, lmList):
    for i in range(4, 17, 4):
        if findDistance(detector, lmList[i], lmList[i+4])[0] < 50:
            return False
    return True


def isOK(detector, lmList):
    if findDistance(detector, lmList[4], lmList[8])[0] < 50:
        if findDistance(detector, lmList[8], lmList[12])[0] > 100:
            if findDistance(detector, lmList[12], lmList[16])[0] > 50:
                if findDistance(detector, lmList[16], lmList[20])[0] > 50:
                    return True
    return False


def clearCanvas(width, height, size):
    canvas = np.zeros((height, width, 3), dtype="uint8")
    x1 = (width - size) // 2
    y1 = 100
    x2 = x1 + size
    y2 = y1 + size
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 0, 0), 3)
    return canvas


def main():
    cap = cv2.VideoCapture(0+cv2.CAP_DSHOW)
    cap.set(3, 1024)
    cap.set(4, 768)
    ret, img = cap.read()
    if ret:
        canvas = clearCanvas(img.shape[1], img.shape[0], 300)
    detector = HandDetector(detectionCon=0.5, maxHands=1)
    prev = (-1, -1)
    while True:
        ret, img = cap.read()
        if not ret:
            continue
        img = cv2.flip(img, 1)
        hands, img = detector.findHands(img, flipType=False)
        if hands:
            lmList = hands[0]['lmList']
            #print(lmList)
            x, y, _ = lmList[8]
            #cv2.circle(canvas, (x, y), 10, (255, 255, 255), cv2.FILLED)
            p1 = lmList[4][0], lmList[4][1]
            p2 = lmList[12][0], lmList[12][1]
            p3 = lmList[8][0], lmList[12][1]
            distance, _, img = findDistance(detector, lmList[4], lmList[12], img)
            index, _, = findDistance(detector, lmList[8], lmList[12])
            if isFist(detector, lmList):
                canvas = clearCanvas(img.shape[1], img.shape[0], 300)
                prev = (-1, -1)
            elif isOK(detector, lmList):
                print("OK")
            elif isBao(detector, lmList):
                print('Bao')
                size = 280
                x = (canvas.shape[1] - size)//2
                y = 110
                cropped = cv2.cvtColor(canvas[y:y+size, x:x+size, :], cv2.COLOR_BGR2GRAY)
                cropped = cv2.resize(cropped, (28, 28))
                #cv2.imshow('cropped', cropped)
                cropped = cropped.astype('float32') / 255
                cropped = np.expand_dims(cropped, -1)
                cropped = np.expand_dims(cropped, 0)
            elif distance < 100 and index > 100:
                if prev == (-1, -1):
                    cv2.circle(canvas, (x,y), 10, (255, 255, 255), cv2.FILLED)
                    prev = (x, y)
                else:
                    cv2.line(canvas, prev, (x, y), (255,255,255), 12)
                    prev = (x, y)
            else:
                prev = (-1, -1)

        img = cv2.add(img, canvas)
        cv2.imshow("Image", img)
        cv2.imshow("Canvas", canvas)
        key = cv2.waitKey(1)
        if key == 27:
            break


if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
