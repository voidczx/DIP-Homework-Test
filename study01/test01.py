# _*_ coding utf-8 _*_
# Author：94342
# Time：  2020/9/716:59
# File：  test01.py
# Engine：PyCharm

import cv2 as cv

img = cv.imread("C:/Users/94342/Pictures/Camera Roll/81122289_p3.jpg")
print(type(img))
cv.namedWindow("aWindow", cv.WINDOW_AUTOSIZE)
cv.imshow("aWindow", img)
cv.waitKey(0)
cv.destroyAllWindows()