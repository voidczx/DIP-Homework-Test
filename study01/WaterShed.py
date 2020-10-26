# _*_ coding utf-8 _*_
# Author：94342
# Time：  2020/9/2311:35
# File：  WaterShed.py
# Engine：PyCharm


import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def main():
    image = cv.imread(r'C:\Users\94342\Pictures\Camera Roll\coin.jpg')
    img = image.copy()[:307, :, :]
    WaterShed(img)
    key = cv.waitKey(0)
    if key & 0xff == ord('q'):
        cv.destroyAllWindows()


def WaterShed(image):
    sharp = np.array([[-1, -1, -1],
                      [-1, 8.9, -1],
                      [-1, -1, -1]])
    #blur = cv.GaussianBlur(image, (9, 9), 0)
    blur = cv.bilateralFilter(image, 5, 150, 100)
    gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)
    cv.imshow('gray', gray)
    ret, binary = cv.threshold(gray, 238, 255, cv.THRESH_BINARY_INV)
    cv.imshow('binary', binary)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    open = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel, iterations=2)
    sureBackGround = cv.dilate(open, kernel, iterations=3)
    distance = cv.distanceTransform(open, cv.DIST_L2, 5)
    #显示距离变化后的结果需要一步normalize
    distanceOutput = distance
    cv.normalize(distance, distanceOutput, 0, 1.0, cv.NORM_MINMAX)
    ret, sureFrontGround = cv.threshold(distance, distance.max()*0.95, 255, cv.THRESH_BINARY)
    #这一步必须有
    sureFrontGround = np.uint8(sureFrontGround)
    unknown = cv.subtract(sureBackGround, sureFrontGround)
    ret, makers = cv.connectedComponents(sureFrontGround)
    makers = makers + 1
    makers[unknown == 255] = 0
    waterShed = cv.watershed(image, makers)
    image[waterShed == -1] = (0, 0, 255)
    cv.imshow('reslut', image)


def FindThreshold(image):
    def TrackBar(threshold):
        ret, binary = cv.threshold(image, threshold, 255, cv.THRESH_BINARY_INV)
        cv.imshow('Binary', binary)
    threshold = 0
    maxThreshold = 255
    cv.namedWindow('Binary', cv.WINDOW_AUTOSIZE)
    cv.createTrackbar('thresholdValue', 'Binary', threshold, maxThreshold, TrackBar)
    cv.imshow('Binary', image)


def FindCanny(image):
    def TrackBar(lowValue):
        canny = cv.Canny(image, lowValue, lowValue*ratio)
        cv.imshow('Canny', canny)
    lowValue = 0
    maxValue = 200
    ratio = 3
    cv.namedWindow('Canny')
    cv.createTrackbar('lowValue', 'Canny', lowValue, maxValue, TrackBar)
    cv.imshow('Canny', image)


if __name__ == '__main__':
    main()
