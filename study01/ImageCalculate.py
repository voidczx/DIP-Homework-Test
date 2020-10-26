# _*_ coding utf-8 _*_
# Author：94342
# Time：  2020/9/923:31
# File：  ImageCalculate.py
# Engine：PyCharm

import cv2 as cv
import numpy as np


def main():
    imageBW = cv.imread('C:/Users/94342/Pictures/Camera Roll/BWimg.jpg')
    imageColorful = cv.imread('C:/Users/94342/Pictures/Camera Roll/rainbow.jpg')
    cv.imshow('origin', imageColorful)
    newImageColorful = ChangeImageContrastAndBrightness(imageColorful, 0.8, -10)
    cv.imshow('new', newImageColorful)
    cv.waitKey(0)


def ChangeImageContrastAndBrightness(image, contrast, brightness):
    h, w, ch = image.shape
    blank = np.zeros([h, w, ch], image.dtype)
    image = cv.addWeighted(image, contrast, blank, 1-contrast, brightness)
    return image


def ImageAdd(img01, img02):
    image = cv.add(img01, img02)
    return image


def ImageSubtract(img01, img02):
    image = cv.subtract(img01, img02)
    return image


def ImageDivide(img01, img02):
    image = cv.divide(img01, img02)
    return image


def ImageMultiply(img01, img02):
    image = cv.multiply(img01, img02)
    return image


def MaskMe():
    capture = cv.VideoCapture(0)
    while(True):
        ret, frame = capture.read()
        if not ret:
            break
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        lowerParam = np.array([156, 43, 46])
        upperPrame = np.array([180, 255, 255])
        mask = cv.inRange(hsv, lowerParam, upperPrame)
        cv.imshow('origin', frame)
        cv.imshow('mask', mask)
        key = cv.waitKey(50)
        if key & 0xff == ord('q'):
            break


def WaitKey(image = None):
    key = cv.waitKey(0)
    if key & 0xff == ord('s'):
        cv.imwrite('new.jpg', image)
        cv.destroyAllWindows()
    elif key & 0xff == ord('q'):
        cv.destroyAllWindows()


def ChangeImageColor(image):
    grey = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    inverse = cv.bitwise_not(image)
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    yuv = cv.cvtColor(image, cv.COLOR_BGR2YUV)
    yCrCb = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
    return hsv


if __name__ == '__main__':
    main()
