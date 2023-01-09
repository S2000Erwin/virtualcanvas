import cv2
import numpy as np


def clear_canvas(width, height, size):
    canvas = np.zeros((height, width, 3), dtype='uint8')
    x1 = (width-size)//2
    y1 = (height-size)//2
    x2 = x1 + size
    y2 = y1 + size
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 0, 0), 3)
    return canvas


def main():
    print(cv2.version)
    cap = cv2.VideoCapture(0+cv2.CAP_DSHOW)
    ret, img = cap.read()
    if not ret:
        return
    canvas = clear_canvas(img.shape[1], img.shape[0], 300)
    while True:
        ret, img = cap.read()
        if not ret:
            break
        img = cv2.flip(img, 1)
        img = cv2.add(img, canvas)
        cv2.imshow('Virtual Canvas', img)
        key = cv2.waitKey(1)
        if key == ord("q") or key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
