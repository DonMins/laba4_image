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
    def simpleGradient(img):
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
    def ConsistentMethod(S, N1, N2, M1, M2):
        def getVecG(x1, x2):
            return np.transpose([x1**2,x2**2,x1*x2,x1, x2, 1])

        indN = 1 - N1
        indM = 1 - M1
        DiapN = range(N1,N2+1)
        DiapM = range(M1,M2+1)
        G = np.zeros((S, S))

        for k in range(0,S):
            for n in DiapN:
                for m in DiapM:
                    G[k,:] = G[k,:]+(getVecG(n, m))[k] * np.transpose(getVecG(n, m))

        Ginv = np.linalg.inv(G)
        F = np.zeros(((N2 + indN, M2 + indM, S)))
        for k in range(0,S):
            for n in DiapN:
                for m in DiapM:
                    for l in range (0,S):
                        F[n + indN - 1, m + indM - 1, k] = F[n + indN - 1, m + indM - 1, k] + Ginv[k, l] * \
                                                           (getVecG(n, m))[l]
        return F

    @staticmethod
    def Convolution(image, kernel):
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


    sgm = cv3.simpleGradient(img)
    vals = sgm.flatten()
    plt.hist(vals, bins=range(256))
    plt.show()

    result = cv3.threshold_processing(sgm, 4)
    cv2.imshow("Simple gradient method", result)

    kernel = np.array([
        [0,1,0],
        [1,-4,1],
        [0,1,0] ])

    res = cv3.Convolution(img, kernel)

    vals = res.flatten()
    plt.hist(vals, bins=range(256))
    plt.show()

    result = cv3.threshold_processing(res, 4)


    cv2.imshow("Laplace Method", result)

    f = cv3.ConsistentMethod(6,-2,2,-2,2)


    a = (f[:,:,0])
    b = (f[:,:,1])

    Fv= 2*a + 2*b
    print(Fv)
    res = cv3.Convolution(img, Fv)
    result = cv3.threshold_processing(res, 4)
    cv2.imshow("Approval Method", result)

    cv2.waitKey(0)
