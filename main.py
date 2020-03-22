import cv2
import numpy as np
import matplotlib.pyplot as plt

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


    @staticmethod
    def Laplace(image, kernel):
        heightM, widthM = kernel.shape
        height, width = image.shape

        centerHeightM = int(np.ceil(heightM / 2) - 1)
        centerwidthM = int(np.ceil(widthM / 2) - 1)

        img2 = np.zeros((height + centerHeightM * 2, width + centerwidthM * 2), np.uint8)
        img2[centerHeightM:height + centerHeightM, centerwidthM:width + centerwidthM] = image

        for i in range(centerHeightM):
            img2[i, :] = img2[centerHeightM, :]
            img2[height + 1 + i, :] = img2[height, :]

        for i in range(centerwidthM):
            img2[:, i] = img2[:, centerwidthM]
            img2[:, width + 1 + i] = img2[:, width]

        new_image = np.zeros((height + centerHeightM, width + centerwidthM))

        for i in range(centerHeightM, height + centerHeightM):
            for j in range(centerwidthM, width + centerwidthM):
                new_image[i - centerHeightM][j - centerwidthM] = np.sum(img2[i - centerHeightM: i + centerHeightM + 1,
                                                                        j - centerwidthM: j + centerwidthM + 1] * kernel)

        return new_image




if __name__ == '__main__':

    img = cv2.imread("rab2.jpg", 0)
    cv2.imshow("input", img)

    result = cv3.threshold_processing(cv3.contour(img), 5)
    cv2.imshow("result", result)

    kernel =np.array([
        [0,0,1,0,0],
        [0,1,2,1,0],
        [1,2,-17,2,1],
        [0,1,2,1,0],
        [0,0,1,0,0]])

    # kernel = 1/2* np.array([
    #     #     [1,0,1,],
    #     #     [0,-4,0],
    #     #     [1,0,1] ])

    kernel = np.array([
        [0,1,0],
        [1,-4,1],
        [0,1,0] ])

    res = cv3.Laplace(img,kernel)


    vals = res.flatten()
    plt.hist(vals, bins=range(256))
    plt.show()

    result = cv3.threshold_processing(res, 5)


    cv2.imshow("result", result)

    cv2.waitKey(0)
