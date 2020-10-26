# _*_ coding utf-8 _*_
# Author：94342
# Time：  2020/9/1413:55
# File：  Hist&HistProcess.py
# Engine：PyCharm
import cv2 as cv
import numpy as np
from matplotlib import pyplot as pt


def main():
    image = cv.imread(r'C:\Users\94342\Pictures\Camera Roll\rainbow.jpg')
    image2 = cv.imread(r'C:\Users\94342\Pictures\Camera Roll\rainbowSample.jpg')
    cv.imshow('origin', image)
    cv.imshow('sample', image2)
    HistCompare(image, image2)
    cv.waitKey(0)


def HistStretch(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imshow('gray', gray)
    hist = gray.ravel()
    theMin = hist.min()
    theMax = hist.max()
    dst = np.uint8(255/(theMax - theMin) * (gray - theMin) + 0.5)
    cv.imshow('stretch', dst)


def ColorfulChoose(target, mask):
    choose = cv.bitwise_and(target, target, mask=mask)
    cv.imshow('choose', choose)


def HistBackProcess(sample, target):
    sampleHSV = cv.cvtColor(sample, cv.COLOR_BGR2HSV)
    targetHSV = cv.cvtColor(target, cv.COLOR_BGR2HSV)
    sampleHist = cv.calcHist([sampleHSV], [0, 1], None, [12, 16], [0, 180, 0, 256])
    cv.normalize(sampleHist, sampleHist, 0, 255, cv.NORM_MINMAX)
    image = cv.calcBackProject([targetHSV], [0, 1], sampleHist, [0, 180, 0, 256], 1)
    ColorfulChoose(target, image)
    cv.imshow('back', image)


def CreatImageHist(image):
    h, w, ch = image.shape
    hist = np.zeros([256*3, 1], np.float32)
    for height in range(h):
        for width in range(w):
            for channel in range(ch):
                index = image[height, width, channel] + 256 * channel
                hist[index, 0] = hist[index, 0] + 1
    return hist


def HistCompare(image1, image2):
    hist1 = CreatImageHist(image1)
    cv.normalize(hist1, hist1, 0, 1, cv.NORM_MINMAX)
    print(hist1.shape)
    print(type(hist1))
    hist2 = CreatImageHist(image2)
    cv.normalize(hist2, hist2, 0, 1, cv.NORM_MINMAX)
    bhatta = cv.compareHist(hist1, hist2, cv.HISTCMP_BHATTACHARYYA)
    correl = cv.compareHist(hist1, hist2, cv.HISTCMP_CORREL)
    chisq = cv.compareHist(hist1, hist2, cv.HISTCMP_CHISQR)
    print('bhatta = %s, correl = %s, chisq = %s' % (bhatta, correl, chisq))


def ImageColorfulHist(image):
    colors = ['blue', 'green', 'red']
    for i, color in enumerate(colors):
        line = cv.calcHist([image], [i], None, [256], [0, 256])
        print(line)
        pt.plot(line, color=color)
        pt.xlim([0, 256])
    pt.show()


#限制对比度的基础上作直方图均衡（使均衡后的图像不会太亮）
def Clahe(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blue = image[:, :, 0]
    green = image[:, :, 1]
    red = image[:, :, 2]
    clahe = cv.createCLAHE(clipLimit=2, tileGridSize=(2, 2))
    image[:, :, 0] = clahe.apply(blue)
    image[:, :, 1] = clahe.apply(green)
    image[:, :, 2] = clahe.apply(red)
    cv.imshow('clahe', image)
    #cv.imshow('clahe', clahe.apply(gray))


def ImageEqualHist(image):
    #image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    #cv.imshow('grey', image)
    #image[:, :, 0] = cv.equalizeHist(image[:, :, 0])
    image[:, :, 1] = cv.equalizeHist(image[:, :, 1])
    image[:, :, 2] = cv.equalizeHist(image[:, :, 2])
    cv.imshow('equal', image)


def ImageHist(image):
    pt.hist(image.ravel(), 256, [0, 256])
    pt.show()


if __name__ == '__main__':
    main()
