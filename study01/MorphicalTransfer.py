# _*_ coding utf-8 _*_
# Author：94342
# Time：  2020/9/2219:44
# File：  MorphicalTransfer.py
# Engine：PyCharm


import cv2 as cv
import numpy as np


def main():
    def ChangedBinary(thresholdValue):
        ret, binary = cv.threshold(gray, thresholdValue, 255, cv.THRESH_BINARY)
        cv.imshow('ChangedBinary', binary)
    thresholdValue = 0
    image = cv.imread(r'C:\Users\94342\Pictures\Camera Roll\coin.jpg')
    image02 = cv.imread(r'C:\Users\94342\Pictures\Camera Roll\81122289_p4.jpg')
    img = image02.copy()
    cv.imshow('origin', img)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 250, 255, cv.THRESH_BINARY_INV)
    cv.imshow('binary', binary)
    #cv.namedWindow('ChangedBinary', cv.WINDOW_AUTOSIZE)
    #cv.createTrackbar('ThresholdValue', 'ChangedBinary', thresholdValue, 255, ChangedBinary)
    #erode = Erode(binary)
    #cv.imshow('erode', erode)
    #dilate = Dilate(binary)
    #cv.imshow('dilate', dilate)
    dst = binary.copy()
    originBinary = binary.copy()
    cv.namedWindow('dst', cv.WINDOW_AUTOSIZE)
    cv.imshow('dst', dst)
    erodeCount = 0
    dilateCount = 0
    while(True):
        key = cv.waitKey(0)
        if key & 0xff == ord('e'):
            dst = Erode(dst)
            erodeCount = erodeCount + 1
        elif key & 0xff == ord('d'):
            dst = Dilate(dst)
            dilateCount = dilateCount + 1
        elif key & 0xff == ord('b'):
            dst = originBinary.copy()
            erodeCount = 0
            dilateCount = 0
        elif key & 0xff == ord('o'):
            dst = Open(dst)
        elif key & 0xff == ord('c'):
            dst = Close(dst)
        elif key & 0xff == ord('h'):
            dst = HorizontalClear(dst)
        elif key & 0xff == ord('v'):
            dst = VerticalClear(dst)
        elif key & 0xff == ord('t'):
            dst = DisturbClear(dst)
        elif key & 0xff == ord('1'):
            dst = InsideGradient(dst)
        elif key & 0xff == ord('2'):
            dst = OutsideGradient(dst)
        elif key & 0xff == ord('q'):
            cv.destroyAllWindows()
            break
        cv.imshow('dst', dst)
    print('Erode Count = %s; Dilate Count = %s' % (erodeCount, dilateCount))


#getStructuringElement中的rect模式相当于numpy.ones
def Erode(binary):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    dst = cv.erode(binary, kernel)
    return dst


def Dilate(binary):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    dst = cv.dilate(binary, kernel)
    return dst

#开操作留黑去白（去除白色噪点）开操作是先腐蚀再膨胀
def Open(binary):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    dst = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
    return dst

#闭操作留白去黑（增大白色轮廓）闭操作是先膨胀再腐蚀
def Close(binary):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    dst = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)
    return dst


#水平、垂直、杂线（杂点）的清楚可依赖开操作
def HorizontalClear(binary):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 1))
    dst = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
    return dst


def VerticalClear(binary):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 15))
    dst = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
    return dst


def DisturbClear(binary):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (9, 9))
    dst = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
    return dst


#顶帽操作（TopHat）是用原图像与开操作相减 效果类似于沙画（在二值图像的基础上）
def TopHat(binary):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    dst = cv.morphologyEx(binary, cv.MORPH_TOPHAT, kernel)
    return dst


def TopHat01(binary):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    open = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
    dst = binary - open
    return dst


#黑帽操作是闭操作与原图像相减，效果类似于珐琅花雕
def BlackHat(binary):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    dst = cv.morphologyEx(binary, cv.MORPH_BLACKHAT, kernel)
    return dst


def BlackHat01(binary):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    close = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)
    dst = close - binary
    return dst


#基本梯度是图像的膨胀减去图像的腐蚀
def BasicGradient(binary):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    dst = cv.morphologyEx(binary, cv.MORPH_GRADIENT, kernel)
    return dst


def BasicGradient01(binary):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    dilate = cv.dilate(binary, kernel)
    erode = cv.erode(binary, kernel)
    dst = dilate - erode
    return dst


#内外梯度交替进行出现有趣的效果 主要是外梯度
#内梯度指图像的原图减去腐蚀
def InsideGradient(binary):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    erode = cv.erode(binary, kernel)
    dst = cv.subtract(binary, erode)
    return dst


#外梯度指膨胀减去原图
def OutsideGradient(binary):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    dilate = cv.dilate(binary, kernel)
    dst = cv.subtract(dilate, binary)
    return dst


if __name__ == '__main__':
    main()