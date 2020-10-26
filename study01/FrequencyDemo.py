# _*_ coding utf-8 _*_
# Author：94342
# Time：  2020/10/622:23
# File：  FrequencyDemo.py
# Engine：PyCharm


import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


if __name__ == '__main__':

    kernelSize = 10
    image = cv.imread('./face.jpg')
    img = image.copy()

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    plt.subplot(221)
    plt.title('origin')
    plt.imshow(gray, cmap='gray')
    row, column = gray.shape
    centerRow, centerColumn = row/2, column/2

    grayDown = gray.copy()
    grayUp = gray.copy()
    # cv.INTER_AREA用于降采样的抽取，可提供无莫尔条纹的效果
    grayDown = cv.resize(grayDown, (column//10, row//10))
    # cv.INTER_LINER线性插值 cv.INTER_CUBIC双线性插值 cv.INTER_NEAREST就近插值
    grayUp = cv.resize(grayUp, (column*2, row*2), interpolation=cv.INTER_CUBIC)

    fft = np.fft.fft2(gray)     #这一步所得结果是复数
    shiftFFT = np.fft.fftshift(fft)     #这一步平移使低频到图像中央
    spectrum = 20 * np.log(np.abs(shiftFFT))

    fftFilter = shiftFFT.copy()

    #频域高通滤波
    # fftFilter[int(centerRow-kernelSize): int(centerRow+kernelSize), int(centerColumn-kernelSize):int(centerColumn+kernelSize)] = 0
    for r in range(int(row)):
        for c in range(int(column)):
            if (np.square(r - centerRow) + np.square(c - centerColumn)) >= np.square(kernelSize):
                fftFilter[r, c] = 0
    spectrumFilter = 20*np.log(np.abs(fftFilter) + np.finfo(np.float32).eps)

    fftFilter = np.fft.ifftshift(fftFilter)
    back = np.fft.ifft2(fftFilter)
    back = np.abs(back)

    backdown = cv.resize(back, dsize=(column//10, row//10))

    plt.figure(1)
    plt.subplot(222), plt.imshow(spectrum, cmap='gray'), plt.title('fft')
    plt.subplot(223), plt.imshow(spectrumFilter, cmap='gray'), plt.title('filter')
    plt.subplot(224), plt.imshow(back, cmap='gray'), plt.title('back')

    plt.figure(2)
    plt.subplot(211), plt.imshow(gray, cmap='gray')
    plt.subplot(212), plt.imshow(grayDown, cmap='gray')

    plt.figure(3)
    plt.subplot(211), plt.imshow(gray, cmap='gray')
    plt.subplot(212), plt.imshow(backdown, cmap='gray')

    cv.imshow('up', grayUp)

    plt.show()


