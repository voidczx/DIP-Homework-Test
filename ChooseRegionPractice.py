# _*_ coding utf-8 _*_
# Author：94342
# Time：  2020/9/1713:37
# File：  ChooseRegionPractice.py
# Engine：PyCharm


import cv2 as cv
import numpy as np


class Point:
    x = 0
    y = 0

    def __init__(self, x, y):
        self.x = x
        self.y = y


class ImageChoose:
    reverseROI = None
    dst = None
    reverseDST = None
    height = 0
    width = 0
    channel = 0
    points = []
    roi = np.zeros([1, 1, 1], np.uint8)

    def __init__(self, image):
        self.origin = image.copy()
        self.image = image
        self.height = image.shape[0]
        self.width = image.shape[1]
        self.channel = image.shape[2]
        self.roi = np.zeros(image.shape, np.uint8)

    def AppendPoint(self, point):
        if self.CheckEqualPoint(point):
            self.DrawPolygon()
        else:
            self.points.append(np.array([point.x, point.y], dtype=np.int32))

    def DrawPolygon(self):
        self.points = np.array(self.points, dtype=np.int32)
        cv.fillPoly(self.roi, [self.points], (255, 255, 255))
        return self.roi

    def ChooseRegion(self):
        self.dst = cv.bitwise_and(self.origin, self.roi)
        return self.dst

    def ChooseEnd(self):
        self.DrawPolygon()
        self.ChooseRegion()
        self.RerverseROI()
        self.ReverseRegion()

    def RerverseROI(self):
        self.reverseROI = cv.bitwise_not(self.roi)
        return self.reverseROI

    def ReverseRegion(self):
        self.reverseDST = cv.bitwise_and(self.reverseROI, self.origin)
        return self.reverseDST

    def CheckEqualPoint(self, point):
        for p in self.points:
            if point.x == p[0] and point.y == p[1]:
                return True
        return False


def main():
    image = cv.imread(r'C:\Users\94342\Pictures\Camera Roll\rainbow.jpg')
    image = cv.GaussianBlur(image, (3, 3), 0)
    core = ImageChoose(image)
    cv.namedWindow('image', cv.WINDOW_AUTOSIZE)
    cv.setMouseCallback('image', MouseClickEvent, param=core)
    cv.imshow('image', image)
    key = cv.waitKey(0)
    if key & 0xff == ord('b'):
        core.ChooseEnd()
        cv.imshow('roi', core.roi)
        cv.imshow('result', core.dst)
        #sharpKernel = np.array([[-1, 0, 0], [-1, 5, -1], [0, 0, -1]], np.float32)
        #sharp = cv.filter2D(core.dst, -1, sharpKernel)
        blur = cv.GaussianBlur(core.reverseDST, (3, 3), 0)
        cv.imshow('blur', blur)
        result = cv.add(blur, core.dst)
        cv.imshow('blurResult', result)
    elif key & 0xff == ord('q'):
        cv.destroyWindow('image')
    cv.waitKey(0)
    cv.destroyAllWindows()


def MouseClickEvent(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        point = Point(x, y)
        param.AppendPoint(point)
        cv.circle(param.image, (x, y), 1, (255, 0, 0), thickness=-1)
        cv.imshow('image', param.image)


if __name__ == '__main__':
    main()