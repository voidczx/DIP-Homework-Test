# _*_ coding utf-8 _*_
# Author：94342
# Time：  2020/9/1510:05
# File：  Binary&TemplateFInd.py
# Engine：PyCharm


import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def main():
    image = cv.imread(r'C:\Users\94342\Pictures\Camera Roll\rainbow.jpg')
    image2 = cv.imread(r'C:\Users\94342\Pictures\Camera Roll\8111111.jpg')
    LargeBinary(image)
    cv.waitKey(0)


def LargeBinary(image):
    hStep = 256
    wStep = 256
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imshow('gray', gray)
    for height in range(0, gray.shape[0], hStep):
        for width in range(0, gray.shape[1], wStep):
            ROI = gray[height:height+hStep, width:width+wStep]
            gray[height:height+hStep, width:width+wStep] = cv.adaptiveThreshold(ROI, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 13, 5)
    cv.imshow('binary', gray)


def AdaptThreshold(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    binary = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 13, 5)
    cv.imshow('adaptBinary', binary)


def ThresholdBinary(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    print(ret)
    cv.imshow('binary', binary)


def TemplateMatched(target, template):
    index = 1
    methods = [cv.TM_CCOEFF_NORMED, cv.TM_CCORR_NORMED, cv.TM_SQDIFF_NORMED, cv.TM_CCOEFF, cv.TM_CCORR, cv.TM_SQDIFF]
    for method in methods:
        result = cv.matchTemplate(target, template, method)
        minValue, maxValue, minLocate, maxLocate = cv.minMaxLoc(result)
        if method == cv.TM_SQDIFF or method == cv.TM_SQDIFF_NORMED:
            startPoint = minLocate
        else:
            startPoint = maxLocate
        height, width = template.shape[0:2]
        endPoint = (startPoint[0]+width, startPoint[1]+height)
        cv.rectangle(target, startPoint, endPoint, (0, 0, 255), 2)
        cv.imshow('method=%s' % index, target)
        index = index + 1


def VisualColorfulHist(image):
    colors = ['blue', 'green', 'red']
    for i, color in enumerate(colors):
        line = cv.calcHist([image], [i], None, [256], [0, 255])
        plt.plot(line, color=color)
    plt.xlim([0, 255])
    plt.show()


if __name__ == '__main__':
    main()
