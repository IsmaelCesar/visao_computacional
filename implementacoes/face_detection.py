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

def hair_detection(image,H,I,ArcCoss,NormRGB):
    global rows, cols
    kImage = image.copy()
    mImage = kImage.copy() / 255
    mImage = np.array([[NormRGB(mImage[i][j][2], mImage[i][j][1], mImage[i][j][0]) for j in range(cols)] for i in range(rows)])
    rImage = mImage[:, :, 2]
    gImage = mImage[:, :, 1]
    bImage = mImage[:, :, 0]


    angles = ArcCoss(rImage,gImage,bImage)
    applyH = np.array([[H(bImage[i][j],gImage[i][j],angles[i][j]) for j in range(cols)] for i in range(rows)])
    #applyI = I(rImage,gImage,bImage)
    applyI = I(kImage[:,:,2], kImage[:,:,1], kImage[:,:,0])
    # Computing B - G and B - R
    #bmg = bImage - gImage
    #bmr = bImage - rImage
    bmr = kImage[:, :, 0] - kImage[:, :, 2]
    bmg = kImage[:,:,0] - kImage[:,:,1]

    for i in range(rows):
        for j in range(cols):
            condition = (applyI[i,j] < 80)
            condition = condition and (bmg[i,j] < 15 or bmr[i,j] < 15)
            condition = condition or (applyH[i,j] > np.pi/3 and applyH[i,j] <= 1/3*np.pi )#2/9*np.pi
            if not condition:
                kImage[i, j, 0] = 0.
                kImage[i, j, 1] = 0.
                kImage[i, j, 2] = 0.

    return kImage

def skin_detection(image,F1,F2,White,H,ArcCos,NormRGB):
    global rows, cols
    kImage = image.copy()
    mImage = kImage.copy()/255
    mImage = np.array([[NormRGB(mImage[i][j][2], mImage[i][j][1], mImage[i][j][0]) for j in range(cols)] for i in range(rows)])
    rImage = mImage[:, :, 2]
    gImage = mImage[:, :, 1]
    bImage = mImage[:, :, 0]

    applyF1 = F1(rImage)
    applyF2 = F2(rImage)

    applyW = White(rImage, gImage)

    # calculating values for hsiModel
    #angles = ArcCos(kImage[:,:,2], kImage[:,:,1], kImage[:,:,0])
    angles = ArcCos(rImage, gImage, bImage)
    #applyH = np.array([[H(kImage[i,j,0], kImage[i,j,1], angles[i][j]) for j in range(cols)] for i in range(rows)])
    applyH = np.array([[H(bImage[i,j], gImage[i,j], angles[i][j]) for j in range(cols)] for i in range(rows)])
    #detection
    for i in range(rows):
        for j in range(cols):
            condition = (gImage[i][j] < applyF1[i][j])
            condition = condition and (gImage[i][j] > applyF2[i][j])

            condition = condition and (applyW[i][j] > 0.001)
            condition = condition and (applyH[i][j] > 4/3*np.pi or applyH[i][j] <= np.pi/4)
            if not condition:
                kImage[i, j, 0] = 0.
                kImage[i, j, 1] = 0.
                kImage[i, j, 2] = 0.

    return kImage

def getNormRGB():
    return lambda r,g,b: [b/(r+g+b+.00000001),g/(r+g+b+.00000001),r/(r+g+b+.00000001)]

def getF1():
    return lambda eta : -1.376*np.power(eta,2) + 1.0743*eta + 0.2

def getF2():
    return lambda epsilon : -0.776*np.power(epsilon,2) + 0.5601*epsilon + 0.18

def whiteRange():
    return lambda r,g: np.power(r - 0.33,2) + np.power(g - 0.33,2)

#Functions for the HSI model
def getArccosFunction():
    return lambda r,g,b: np.arccos(0.5 * ((r - g) + (r - b)) / np.sqrt(np.power(r - g, 2) + (r-b)*(g-b)+.00000001))

def getH():
    return lambda B, G, theta: theta if B <= G else 2*np.pi - theta if B > G else 0

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
    NormRGB = getNormRGB()
    I = getI()
    White = whiteRange()

    skinDetect = skin_detection(image,F1,F2,White,H,ArcCos,NormRGB)
    hairDetect = hair_detection(image,H, I, ArcCos, NormRGB)

    cv2.imshow("Original",image)
    cv2.imshow("Detection(Skin)",skinDetect)
    cv2.imshow("Detection(Hair)",hairDetect)
    cv2.waitKey(0)
    # resultI = I(rImage,gImage,bImage)


if __name__ == '__main__':
    main()