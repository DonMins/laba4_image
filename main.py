import cv2
import numpy as np

class cv3:
    def __init__(self):
        pass

    @staticmethod
    def threshold_processing(img, threshold):
        p = img.shape
        width = p[1]
        height = p[0]

        for i in range(height):
            for j in range(width):
                if (img[i, j] > threshold):
                    img[i, j] = 255
                else:
                    img[i, j] = 0
        return img

    @staticmethod
    def contour(img):
        height = img.shape[0]
        width = img.shape[1]

        img2 = np.zeros((height + 1, width + 1))
        img2[1:height + 1, 1:width + 1] = img

        s1 = np.zeros((height, width), np.uint8)
        s2 = np.zeros((height, width), np.uint8)

        for i in range(1, height + 1):
            for j in range(1, width + 1):
                s1[i - 1, j - 1] = img2[i, j] - img2[i - 1, j]
                s2[i - 1, j - 1] = img2[i, j] - img2[i, j - 1]

        g = np.abs(s1) + np.abs(s2)
        return g


if __name__ == '__main__':

    img = cv2.imread("rab2.jpg", 0)
    cv2.imshow("input", img)

    result = cv3.threshold_processing(cv3.contour(img), 195)
    cv2.imshow("result", result)

    cv2.waitKey(0)
