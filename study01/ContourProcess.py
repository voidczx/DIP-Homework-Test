# _*_ coding utf-8 _*_
# Author：94342
# Time：  2020/9/2116:11
# File：  ContourProcess.py
# Engine：PyCharm


import cv2 as cv
import numpy as np


def main():
    image = cv.imread(r'C:\Users\94342\Pictures\Camera Roll\coin.jpg')
    MeasureContours(image)
    key = cv.waitKey(0)
    if key & 0xff == ord('q'):
        cv.destroyAllWindows()


def MeasureContours(image):
    def MeatureAreas(contours):
        areas = []
        for i, contour in enumerate(contours):
            area = cv.contourArea(contour)
            areas.append(area)
        return areas
    def MeatureRectangles(contours):
        rects = []
        for i, contour in enumerate(contours):
            rect = cv.boundingRect(contour)
            rects.append(rect)
        return rects
    def MeatureMoments(contours):
        moments = []
        for i, contour in enumerate(contours):
            moment = cv.moments(contour)
            moments.append(moment)
        return moments
    def MeatureCenter(moment):
        if moment['m00'] != 0:
            centerX = moment['m01'] / ['m00']
            centerY = moment['m10'] / ['m00']
            center = (centerX, centerY)
            return center
        else:
            return None
    def MeatureWidthHeightRatio(rect):
        ratio = min(rect[2], rect[3]) / max(rect[2], rect[3])
        return ratio
    img = image.copy()
    contours = GetContours(img)
    for index, contour in enumerate(contours):
        approxCurve = cv.approxPolyDP(contour, 10, True)
        if approxCurve.shape[0] > 6:
            cv.drawContours(img, contours, index, (0, 255, 255), 2)
        elif approxCurve.shape[0] > 4:
            cv.drawContours(img, contours, index, (255, 0, 255), 2)
        print(approxCurve.shape)
        area = cv.contourArea(contour)
        #boundingRect函数的返回值是（x, y, w, h），xy是矩形左上点，wh是矩形的宽和高
        rect = cv.boundingRect(contour)
        #print(rect)
        widthHeightRatio = min(rect[2], rect[3]) / max(rect[2], rect[3])
        #print('WidthHeightRatio = %s' % widthHeightRatio)
        print('area = %s' % area)
        cv.rectangle(img, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (255, 0, 0))
        moments = cv.moments(contour)
        #print(moments)
        if moments['m00'] != 0:
            xPosition = int(moments['m10'] / moments['m00'])
            yPosition = int(moments['m01'] / moments['m00'])
            cv.circle(img, (xPosition, yPosition), 2, (0, 0, 255), -1)
    cv.imshow('point', img)
    #cv.imshow('rect', img)


def ChangeableFindContours(image):
    def TrackBarUpdata(lowValue):
        img = image.copy()
        blur = cv.GaussianBlur(img, (3, 3), 0)
        canny = cv.Canny(blur, lowValue, lowValue*3)
        contours, hierachy = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        for index, contour in enumerate(contours):
            cv.drawContours(img, contours, index, (255, 0, 0), 1)
        cv.imshow('contoursDetect', img)
    lowValue = 0
    maxValue = 200
    cv.namedWindow('contoursDetect', cv.WINDOW_AUTOSIZE)
    cv.createTrackbar('lowValue', 'contoursDetect', lowValue, maxValue, TrackBarUpdata)
    cv.imshow('contoursDetect', image)


def GetContours(image):
    img = image.copy()
    img = cv.GaussianBlur(img, (3, 3), 0)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #canny = cv.Canny(img, 150, 450)
    #threshold函数有两个返回值(真贱呐)
    ret, binary = cv.threshold(gray, 205, 255, cv.THRESH_BINARY_INV)
    print(ret)
    cv.imshow('canny', binary)
    contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    #print(contours)
    for index, contour in enumerate(contours):
        cv.drawContours(img, contours, index, (0, 0, 255), -1)
    cv.imshow('result', img)
    return contours


def ChangableChanny(image):
    def TrackBarUpdata(lowValue):
        img = image.copy()
        img = cv.GaussianBlur(img, (3, 3), 0)
        ratio = 3
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        canny = cv.Canny(img, lowValue, lowValue*ratio)
        cv.imshow('Canny', canny)
    lowValue = 0
    maxValue = 100
    cv.namedWindow('Canny', cv.WINDOW_AUTOSIZE)
    cv.createTrackbar('lowValue', 'Canny', lowValue, maxValue, TrackBarUpdata)


if __name__ == '__main__':
    main()
