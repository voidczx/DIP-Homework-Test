# _*_ coding utf-8 _*_
# Author：94342
# Time：  2020/9/1214:00
# File：  Fill&Blur.py
# Engine：PyCharm
import cv2 as cv
import numpy as np
import tensorflow as tf


def main():
    #image = cv.imread(r'C:\Users\94342\Pictures\Camera Roll\rainbow.jpg')
    #CustomBlur(image)
    #cv.waitKey(0)
    pass


def FillColorfulImage(image):
    h, w = image.shape[:2]
    mask = np.zeros([h+2, w+2], np.uint8)
    cv.floodFill(image, mask, (300, 300), (255, 0, 0), (50, 50, 50), (20, 20, 20), cv.FLOODFILL_FIXED_RANGE)
    return image


def FillBinaryImage():
    image = np.zeros([800, 800, 3], np.uint8)
    image[301: 501, 301: 501] = 255
    cv.imshow('origin', image)
    mask = np.ones([802, 802, 1], np.uint8)
    mask[301: 501, 301: 501] = 0
    cv.floodFill(image, mask, (400, 400), (255, 255, 0), cv.FLOODFILL_MASK_ONLY)
    cv.imshow('fill', image)


def TestFill(image):
    cv.imshow('origin', image)
    image[100:200, 100:200] = 255
    mask = np.ones([image.shape[0]+2, image.shape[1]+2, 1], np.uint8)
    mask[100:200, 100:200] = 0
    cv.floodFill(image, mask, (150, 150), (0, 255, 0), cv.FLOODFILL_MASK_ONLY)
    cv.imshow('fill', image)


'''均值模糊可以用来去噪(随机噪声)'''


def BlurDemo(image):
    cv.imshow('origin', image)
    image = cv.blur(image, (5, 5))
    cv.imshow('blur', image)


'''中值模糊可以用来去除椒盐噪声'''


def MedianBlurDemo(image):
    cv.imshow('origin', image)
    image = cv.medianBlur(image, 5)
    cv.imshow('blur', image)


def CustomBlur(image):
    cv.imshow('origin', image)
    kernel = np.array([[-1, 0, 0], [-1, 5, -1], [0, 0, -1]], np.float32)
    image = cv.filter2D(image, -1, kernel=kernel)
    cv.imshow('filter', image)


if __name__ == '__main__':
    main()