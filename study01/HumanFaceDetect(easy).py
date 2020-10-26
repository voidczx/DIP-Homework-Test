# _*_ coding utf-8 _*_
# Author：94342
# Time：  2020/9/2417:14
# File：  HumanFaceDetect(easy).py
# Engine：PyCharm


import cv2 as cv


def main():
    image = cv.imread('./new.jpg')
    img = image.copy()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faceDetector = cv.CascadeClassifier(r'E:\pyth38\Lib\site-packages\cv2\data\haarcascade_frontalface_alt2.xml')
    capture = cv.VideoCapture(0)
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        else:
            frame = cv.flip(frame, 2)
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            faces = faceDetector.detectMultiScale(gray, 1.2, 5)
            for x, y, w, h in faces:
                cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv.imshow('me', frame)
            key = cv.waitKey(40)
            if key & 0xff == ord('q'):
                break


if __name__ == '__main__':
    main()