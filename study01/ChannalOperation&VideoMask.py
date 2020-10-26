# _*_ coding utf-8 _*_
# Author：94342
# Time：  2020/9/717:15
# File：  ChannalOperation&VideoMask.py
# Engine：PyCharm

import cv2 as cv
import numpy as nm


def main():
    ChooseByCapture()


def ChooseByCapture():
    capture = cv.VideoCapture(0)
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        lowerParam = nm.array([156, 43, 46])
        upperParam = nm.array([180, 255, 255])
        mask = cv.inRange(hsv, lowerParam, upperParam)
        solve = cv.bitwise_and(frame, frame, mask=mask)
        cv.imshow('origin', frame)
        cv.imshow('mask', mask)
        cv.imshow('solve', solve)
        key = cv.waitKey(50)
        if key & 0xff == ord('q'):
            break


def ColorChannalSplit(image):
    b, g, r = cv.split(image)
    return b, g, r


def ColorChannalMerge(b, g, r):
    image = cv.merge([b, g, r])
    return image


def ChineseImread(src):
    image = cv.imdecode(nm.fromfile(src, dtype=nm.uint8), -1)
    return image


def TwoColorChannal(image):
    noBlue = nm.copy(image)
    noBlue[:, :, 0] = 0
    noGreen = nm.copy(image)
    noGreen[:, :, 1] = 0
    noRed = nm.copy(image)
    noRed[:, :, 2] = 0
    VisualImage('noBlue', noBlue)
    VisualImage('noGreen', noGreen)
    VisualImage('noRed', noRed)


def GenerateNewImage():
    image = nm.zeros([400, 400, 3], nm.uint8)
    image[:, :, :] = nm.ones([400, 400, 3]) * 127
    return image


def VisualImage(name, image):
    cv.namedWindow(name, cv.WINDOW_AUTOSIZE)
    cv.imshow(name, image)
    key = cv.waitKey(0)
    if key & 0xff == ord('s'):
        cv.imwrite('%s.png' % name, image)
        cv.destroyWindow(name)
    elif key & 0xff == ord('q'):
        cv.destroyWindow(name)


def PrintImageInfo(image):
    print("Shape = ", image.shape)
    print("Size = ", image.size)
    print("DType = ", image.dtype)


def ChangeImage(image):
    height = image.shape[0]
    width = image.shape[1]
    channal = image.shape[2]
    for hei in range(0, height):
        for wid in range(0, width):
            for cha in range(0, channal):
                originImage = image[hei, wid, cha]
                image[hei, wid, cha] = 255 - originImage
    return image


def CaptureMe():
    cap = cv.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            frame = cv.flip(frame, 1)
            inverse = cv.bitwise_not(frame)
            cv.imshow("video", frame)
            cv.imshow("greyVideo", grey)
            cv.imshow("inverse", inverse)
            key = cv.waitKey(50)
            if key & 0xff == ord('s'):
                cv.imwrite("new.jpg", frame)
            if key & 0xff == ord('q'):
                cv.destroyWindow("video")
                cv.destroyWindow("greyVideo")
                cv.destroyWindow("inverse")
                break
        else:
            break


if __name__ == "__main__":
    main()




