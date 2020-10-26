# _*_ coding utf-8 _*_
# Author：94342
# Time：  2020/9/1819:11
# File：  HoughDetect.py
# Engine：PyCharm


import cv2 as cv
import numpy as np


def main():
    image = cv.imread(r'C:\Users\94342\Pictures\Camera Roll\rainbow.jpg')
    image01 = image.copy()
    image02 = cv.imread(r'C:\Users\94342\Pictures\Camera Roll\coin.jpg')
    cv.imshow('origin', image02)
    HoughLineDetect01(image02)
    key = cv.waitKey(0)
    if key & 0xff == ord('q'):
        cv.destroyAllWindows()


def HoughCircleDetect01(image):
    excussion = cv.pyrMeanShiftFiltering(image, 10, 30)
    #excussion = image.copy()
    #excussion = cv.GaussianBlur(excussion, (3, 3), 0)
    cv.imshow('excussion', excussion)
    gray = cv.cvtColor(excussion, cv.COLOR_BGR2GRAY)
    cv.imshow('gray', gray)
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 20, param1=100, param2=50)
    #param1为Canny边缘检测的高阈值， param2越大，则检测出的圆形越接近完美圆形
    circles = np.int16(np.around(circles))
    print(circles)
    #将circles中的内容转为整数
    for temp in circles:
        for circle in temp:
            x = circle[0]
            y = circle[1]
            radius = int(circle[2])
            cv.circle(image, (x, y), radius, (0, 0, 255), 2)
    cv.imshow('result', image)


def HoughLineDetect03(image):
    blur = cv.GaussianBlur(image, (3, 3), 0)
    gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)
    edge = cv.Canny(gray, 30, 90, apertureSize=3)
    cv.imshow('edge3', edge)
    lines = cv.HoughLines(edge, 1, np.pi/180, 118)
    for line in lines:
        radius = line[0][0]
        theta = line[0][1]
        sin = np.sin(theta)
        cos = np.cos(theta)
        x0 = radius*cos
        y0 = radius*sin
        x1 = int(x0 + 1000*(-sin))
        y1 = int(y0 + 1000*cos)
        x2 = int(x0 - 1000*(-sin))
        y2 = int(y0 - 1000*cos)
        cv.line(image, (x1, y1), (x2, y2), (255, 255, 255))
    cv.imshow('result3', image)


def HoughLineDetect02(image):
    blur = cv.GaussianBlur(image, (3, 3), 0)
    gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)
    edge = cv.Canny(gray, 25, 75, apertureSize=3)
    cv.imshow('edge2', edge)
    lines = cv.HoughLinesP(edge, 1, np.pi/180, 118, minLineLength=10, maxLineGap=50)
    #minLineLength参数决定了被认为是直线（线段）的最小长度， maxLineGap参数决定了在同一条直线上的两条线段相隔多远会被认为成一条线段
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv.line(image, (x1, y1), (x2, y2), (255, 0, 0))
    cv.imshow('result2', image)


def HoughLineDetect01(image):
    blur = cv.GaussianBlur(image, (3, 3), 0)
    gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)
    edge = cv.Canny(gray, 30, 90, apertureSize=3)
    cv.imshow('edge1', edge)
    lines = cv.HoughLines(edge, 1, np.pi/180, 118)  #118是经验性的值
    print(lines)
    for line in lines:
        radius = line[0][0]
        theta = line[0][1]
        print(radius)
        print(theta)
        if theta < np.pi/4.0 or theta > 3.0*np.pi/4.0:
            start = (int(radius / np.cos(theta)), 0)
            end = (int(start[0] - image.shape[0]*np.sin(theta)/np.cos(theta)), image.shape[0]-1)
        else:
            start = (0, int(radius/np.sin(theta)))
            end = (image.shape[1]-1, int(start[1] - image.shape[1]*np.cos(theta)/np.sin(theta)))
        cv.line(image, start, end, (255, 255, 255))
    cv.imshow('result1', image)


if __name__ == '__main__':
    main()
