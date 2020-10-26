# _*_ coding utf-8 _*_
# Author：94342
# Time：  2020/9/1315:53
# File：  EdgeBlur&Noise.py
# Engine：PyCharm
import cv2 as cv
import numpy as np
import random


def main():
    image = cv.imread(r'C:\Users\94342\Pictures\Camera Roll\rainbow.jpg')
    cv.imshow('origin', image)
    SnowNoise(image, 0.05)
    cv.imshow('blured', cv.medianBlur(image, 5))
    cv.waitKey(0)


#双边滤波，类似于磨皮效果


def BilateralFilter(image):
    image = cv.bilateralFilter(image, 5, 150, 50)
    cv.imshow('bilageralFilter', image)


#偏移滤波，类似于卡通效果


def PyrMeanShiftFiltering(image):
    image = cv.pyrMeanShiftFiltering(image, 10, 30)
    cv.imshow('pyrM', image)


def NumberLimit(num):
    if num > 255:
        return 255
    elif num < 0:
        return 0
    else:
        return num


def GuassianNoise(image):
    h, w, ch = image.shape
    for height in range(h):
        for width in range(w):
            randomNumber = np.random.normal(0, 20, 3)
            for channel in range(ch):
                image[height, width, channel] = NumberLimit(image[height, width, channel] + randomNumber[channel])
    cv.imshow('noise', image)
    return image


def SpicedSaltNoise(image, prob):
    noiseNum = int(image.shape[0]*image.shape[1]*prob)
    for number in range(noiseNum):
        randY = random.randint(0, image.shape[1]-1)
        randX = random.randint(0, image.shape[0]-1)
        randomNum = random.random()
        if randomNum < 0.5:
            image[randX, randY] = 0
        else:
            image[randX, randY] = 255
    cv.imshow('spicedSaltNoise', image)


def SnowNoise(image, prob):
    thre = 1 - prob
    output = np.zeros(image.shape, np.uint8)
    for height in range(image.shape[0]):
        for width in range(image.shape[1]):
            randomNum = random.random()
            if randomNum < prob:
                output[height, width] = 0
            elif randomNum > thre:
                output[height, width] = 255
            else:
                output[height, width] = image[height, width]
    cv.imshow('snowNoise', output)


def GuassianBlur(image):
    image = cv.GaussianBlur(image, (15, 15), 0)
    cv.imshow('guassianBlur', image)


if __name__ == '__main__':
    main()
