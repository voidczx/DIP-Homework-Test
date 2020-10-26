# _*_ coding utf-8 _*_
# Author：94342
# Time：  2020/9/1623:29
# File：  Sobel&Laplace&Canny&Pyramid.py
# Engine：PyCharm


import cv2 as cv
import numpy as np


def main():
    image = cv.imread(r'C:\Users\94342\Pictures\Camera Roll\rainbow.jpg')
    cv.imshow('origin', image)
    img = cv.resize(image, (512, 512))
    cv.imshow('img', img)
    Pyramid(img, 5)
    key = cv.waitKey(0)
    if key & 0xff == ord('q'):
        cv.destroyAllWindows()


def CannyDanymic(image):
    def CannyThreshod(lowThreshod):
        temp = image.copy()
        temp = cv.GaussianBlur(temp, (3, 3), 0)
        gray = cv.cvtColor(temp, cv.COLOR_BGR2GRAY)
        dst = cv.Canny(gray, lowThreshod, lowThreshod*ratio, apertureSize=kernelSize)
        dst = cv.bitwise_and(temp, temp, mask=dst)
        cv.imshow('CannyDemo', dst)
    lowThreshod = 0
    maxThreshod = 100
    ratio = 3
    kernelSize = 3
    cv.namedWindow('CannyDemo')
    cv.createTrackbar('lowThreshod', 'CannyDemo',lowThreshod, maxThreshod, CannyThreshod)


#Canny边缘提取，类似于粉笔画
def CannyDetect(image):
    blur = cv.GaussianBlur(image, (3, 3), 0)
    gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)
    sobelX = cv.Sobel(gray, cv.CV_16SC1, 1, 0)
    sobelY = cv.Sobel(gray, cv.CV_16SC1, 0, 1)
    bilibili = cv.Canny(sobelX, sobelY, 50, 150)
    cv.imshow('bilibili', bilibili)
    cv.imshow('colorfulBi', cv.bitwise_and(image, image, mask=bilibili))
    other = cv.Canny(gray, 50, 150)
    cv.imshow('other', other)
    cv.imshow('colorfulOhter', cv.bitwise_and(image, image, mask=other))


def SobelCalc(image):
    blur = cv.GaussianBlur(image, (3, 3), 0)
    sobelX = cv.Sobel(blur, cv.CV_32F, 1, 0)
    sobelY = cv.Sobel(blur, cv.CV_32F, 0, 1)
    sobelX = cv.convertScaleAbs(sobelX)
    sobelY = cv.convertScaleAbs(sobelY)
    cv.imshow('sobelX', sobelX)
    cv.imshow('sobelY', sobelY)
    sobelDouble = cv.addWeighted(sobelX, 0.5, sobelY, 0.5, 0)
    cv.imshow('sobelDouble', sobelDouble)


def LaplaceCalc(image):
    blur = cv.GaussianBlur(image, (3, 3), 0, 0)
    cv.imshow('blur', blur)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imshow('gray', gray)
    laplace = cv.Laplacian(blur, cv.CV_32F, ksize=3)
    laplace = cv.convertScaleAbs(laplace)
    cv.imshow('laplace', laplace)


def LaplaceCalcEightNeighber(image):
    blur = cv.GaussianBlur(image, (3, 3), 0)
    kernel = np.array([[1, 1, 1],
                       [1, -8, 1],
                       [1, 1, 1]], np.float32)
    dst = cv.filter2D(blur, cv.CV_32F, kernel=kernel)
    result = cv.convertScaleAbs(dst)
    cv.imshow('eightLaplace', result)


def PyramidDown(image, times):
    temp = np.copy(image)
    #temp = cv.GaussianBlur(temp, (3, 3), 0)
    for i in range(times):
        temp = cv.pyrDown(temp)
        cv.imshow(str(i), temp)


def EqualSizeOfTwo(image1, image2):
    times = 0
    if image1.shape[0] > image2.shape[0]:
        times = image1.shape[0] - image2.shape[0]
        for i in range(times):
            image1 = np.delete(image1, image1.shape[0]-1, 0)
    if image1.shape[1] > image2.shape[1]:
        times = image1.shape[1] - image2.shape[1]
        for i in range(times):
            image1 = np.delete(image1, image1.shape[1]-1, 1)
    if image2.shape[0] > image1.shape[0]:
        times = image2.shape[0] - image1.shape[0]
        for i in range(times):
            image2 = np.delete(image2, image2.shape[0]-1, 0)
    if image2.shape[1] > image1.shape[1]:
        times = image2.shape[1] - image1.shape[1]
        for i in range(times):
            image2 = np.delete(image2, image2.shape[1]-1, 1)
    return image1, image2


def PyramidLapulas(image):
    temp = image.copy()
    temp1 = cv.pyrDown(temp)
    temp2 = cv.pyrDown(temp1)
    temp3 = cv.pyrUp(temp2, dstsize=temp1.shape[:2])
    result = cv.subtract(temp3, temp1)
    cv.imshow('result', result)


#需要image的大小是2的n次方
def Pyramid(image, times):
    downImages = []
    temp = image.copy()
    downImages.append(temp)
    for i in range(1, times+1):
        temp = cv.pyrDown(temp)
        downImages.append(temp)
    for i in range(len(downImages)):
        cv.imshow(str(i), downImages[i])
    for i in range(len(downImages)-1, 0, -1):
        lpls = cv.subtract(cv.pyrUp(downImages[i]), downImages[i-1])
        cv.imshow('lpls %s' % i, lpls)


if __name__ == '__main__':
    main()
