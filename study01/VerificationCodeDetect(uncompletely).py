# _*_ coding utf-8 _*_
# Author：94342
# Time：  2020/9/2612:42
# File：  VerificationCodeDetect(uncompletely).py
# Engine：PyCharm


import cv2 as cv
import numpy as np
import pytesseract as tess
from PIL import Image


def main():
    # Image Read

    # image's binary threshold = 128
    image = cv.imread(r'C:\Users\94342\Pictures\Camera Roll\VerificationCode.jpg')
    # image02's binary threshold = 195
    image02 = cv.imread(r'C:\Users\94342\Pictures\Camera Roll\VerificationCode02.jpg')
    # image03's binary threshold = 148
    image03 = cv.imread(r'C:\Users\94342\Pictures\Camera Roll\VerificationCode03.jpg')

    # End


    # Origin Image Show

    # cv.imshow('origin', image)
    # cv.imshow('origin02', image02)
    # cv.imshow('origin03', image03)

    # End


    # Find Image Binary Threshold

    # img = image03.copy()
    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # ChangedBinary(gray)

    # End


    # Blur Images

    blur01 = cv.GaussianBlur(image, (17, 17), 0)
    blur02 = cv.GaussianBlur(image02, (3, 3), 0)
    blur03 = cv.GaussianBlur(image03, (7, 7), 0)

    # End


    # Gray Images

    gray = cv.cvtColor(blur01, cv.COLOR_BGR2GRAY)
    gray02 = cv.cvtColor(blur02, cv.COLOR_BGR2GRAY)
    gray03 = cv.cvtColor(blur03, cv.COLOR_BGR2GRAY)

    # End


    # Binary Images

    ret01, binary01 = cv.threshold(gray, 170, 256, cv.THRESH_BINARY_INV)
    ret02, binary02 = cv.threshold(gray02, 195, 256, cv.THRESH_BINARY_INV)
    ret03, binary03 = cv.threshold(gray03, 145, 256, cv.THRESH_BINARY_INV)

    # End


    #Clear Mess Lines

    kernel01 = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
    kernel02 = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    kernel03 = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
    binary01 = cv.morphologyEx(binary01, cv.MORPH_OPEN, kernel01)
    binary02 = cv.morphologyEx(binary02, cv.MORPH_OPEN, kernel02)
    binary03 = cv.morphologyEx(binary03, cv.MORPH_OPEN, kernel03)
    binary01 = cv.bitwise_not(binary01)
    binary02 = cv.bitwise_not(binary02)
    binary03 = cv.bitwise_not(binary03)
    cv.imshow('binary01', binary01)
    cv.imshow('binary02', binary02)
    cv.imshow('binary03', binary03)
    textImage01 = Image.fromarray(binary01)
    textImage02 = Image.fromarray(binary02)
    textImage03 = Image.fromarray(binary03)

    # End


    #Generate String

    print(tess.image_to_string(textImage01))
    print(tess.image_to_string(textImage02, lang='eng'))
    print(tess.image_to_string(textImage03))

    # Save Images

    key = cv.waitKey(0)
    # if key & 0xff == ord('s'):
    #     cv.imwrite('VerificationCodePro01.png', binary01)
    #     cv.imwrite('VerificationCodePro02.png', binary02)
    #     cv.imwrite('VerificationCodePro03.png', binary03)
    # else:
    #     cv.destroyAllWindows()

    #End


def ChangedBinary(image):
    def TrackBar(thresholdValue):
        ret, binary = cv.threshold(image, thresholdValue, 255, cv.THRESH_BINARY_INV)
        cv.imshow('Binary', binary)
    thresholdValue = 0
    cv.namedWindow('Binary', cv.WINDOW_AUTOSIZE)
    cv.imshow('Binary', image)
    cv.createTrackbar('Threshold Value', 'Binary', thresholdValue, 255, TrackBar)


if __name__ == '__main__':
    main()
