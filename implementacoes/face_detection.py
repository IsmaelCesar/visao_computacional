"""
Implementation of the algorithm:
CHEN, Yao-Jiunn; LIN, Yen-Chun. Simple face-detection algorithm based on minimum facial features.
In: IECON 2007-33rd Annual Conference of the IEEE Industrial Electronics Society. IEEE, 2007. p. 455-460.
"""
import numpy as np
import scipy
import cv2

global images_folder

images_folder = 'imagens/'
image_selected= ['person1.jpg',
                 'person2.jpg',
                 'person3.jpg',
                 'person4.jpg']

global width, height

def hair_detection(image,H,I):
    mImage = image.copy()
    #Normalization
    #mImage = mImage/255
    epsilon = .000000001
    for i in range(mImage.shape[0]):
        for j in range(mImage.shape[1]):
            # Relative frequency
            summation = (mImage[i, j, 0] + mImage[i, j, 1] + mImage[i, j, 2] + epsilon)/255
            r = np.divide(mImage[i, j, 2], summation)
            g = np.divide(mImage[i, j, 1], summation)
            b = np.divide(mImage[i, j, 0], summation)
            # Compute values for the HSI color model
            arg_arccos = 0.5 * ((r - g) + (r - b)) / np.sqrt(np.power(r - g, 2) + np.power(r - b, 2))
            theta = np.arccos(arg_arccos)
            h = H(b,g,theta)
            y = I(r,g,b)
            condition = (y<80 and (b-g < 15 or b-r < 15)) or ( h>20 and h<=40)
            if(not condition):
                mImage[i, j, 0] = 0.
                mImage[i, j, 1] = 0.
                mImage[i, j, 2] = 0.
            else:
                mImage[i, j, 0] = np.multiply(b,mImage[i, j, 0] + mImage[i, j, 1] + mImage[i, j, 2] + epsilon)
                mImage[i, j, 1] = np.multiply(g,mImage[i, j, 0] + mImage[i, j, 1] + mImage[i, j, 2] + epsilon)
                mImage[i, j, 2] = np.multiply(r,mImage[i, j, 0] + mImage[i, j, 1] + mImage[i, j, 2] + epsilon)

    return  mImage


def skin_detection(image,F1,F2,White,H,ArcCos):
    global rows, cols
    kImage = image.copy()
    mImage= kImage / 255

    rImage = mImage[:, :, 2]
    gImage = mImage[:, :, 1]
    bImage = mImage[:, :, 0]

    applyF1 = F1(rImage)
    applyF2 = F2(rImage)

    applyW = White(rImage, gImage)

    # computing g < F1(r)
    compF1 = np.array([[1 if gImage[i][j] < applyF1[i][j] else 0 for j in range(cols)] for i in range(rows)])
    compF2 = np.array([[1 if gImage[i][j] > applyF2[i][j] else 0 for j in range(cols)] for i in range(rows)])
    compW = np.array([[1 if applyW[i][j] > .001 else 0 for j in range(cols)] for i in range(rows)])
    # intersection
    compF1F2W = np.array([[1 if compF1[i][j] == compF2[i][j] else 0 for j in range(cols)] for i in range(rows)])
    compF1F2W = np.array([[1 if compF1F2W[i][j] == compW[i][j] else 0 for j in range(cols)] for i in range(rows)])

    compF1 = None
    compF2 = None
    applyW = None

    # calculating values for hsiModel
    angles = ArcCos(rImage, bImage, bImage)
    applyH = np.array([[H(bImage[i][j], gImage[i][j], angles[i][j]) for j in range(cols)] for i in range(rows)])

    greaterH = np.array([[1 if applyH[i][j] > 240 else 0 for j in range(cols)] for i in range(rows)])
    lesserH = np.array([[1 if applyH[i][j] <= 20 else 0 for j in range(cols)] for i in range(rows)])
    # Union H > 240 U H <= 20
    unionGHLH = greaterH + lesserH

    greaterH = None
    lesserH = None
    detect = np.array([[1  if compF1F2W[i][j] == unionGHLH[i][j] else 0 for j in range(cols)] for i in range(rows)])

    for i in range(rows):
        for j in range(cols):
            if detect[i][j] != 1:
                kImage[i][j][0] = 0.
                kImage[i][j][1] = 0.
                kImage[i][j][2] = 0.

    return kImage

def getF1():
    return lambda eta : -1.37*np.power(eta,2) + 1.0743*eta + 0.2

def getF2():
    return lambda epsilon : -0.776*np.power(epsilon,2) + 0.5601*epsilon + 0.18

def whiteRange():
    return lambda r,g: np.power(r - 0.33,2) + np.power(g - 0.33,2)
#Functions for the HSI model
def getArccosFunction():
    return lambda r,g,b: 0.5 * ((r - g) + (r - b)) / np.sqrt(np.power(r - g, 2) + np.power(r - b, 2))

def getH():
    return lambda B, G, theta: theta if B <= G else 360 - theta if B > G else 0

def getI():
    return lambda R, G, B:  1/3*(G + R + B)






def main():
    global rows, cols
    image = cv2.imread(images_folder + image_selected[2])
    rows = image.shape[0]
    cols = image.shape[1]

    F1 = getF1()
    F2 = getF2()
    H = getH()
    ArcCos = getArccosFunction()
    I = getI()
    White = whiteRange()

    newDetect = skin_detection(image,F1,F2,White,H,ArcCos)

    cv2.imshow("Original",image)
    cv2.imshow("Detection(Skin)",newDetect)
    cv2.waitKey(0)
    # resultI = I(rImage,gImage,bImage)


if __name__ == '__main__':
    main()