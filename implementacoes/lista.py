import  cv2
import  numpy as np

global image_folder
image_folder = "imagens/"

def item1():
    p = cv2.imread(image_folder+"polygons.png")
    pGray = cv2.cvtColor(p,cv2.COLOR_BGR2GRAY)

    ret,thresh = cv2.threshold(pGray,150,255,cv2.THRESH_BINARY_INV)

    im2,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    perimeters = []
    for cnt in contours:
        perimeters.append(cv2.arcLength(cnt,True))

    np.array(perimeters)

    ct = []

    for p,c in zip(perimeters,contours):
        approx = cv2.approxPolyDP(c,0.025*p,True)
        print(len(approx))
        for ap in approx:
            print(ap)
        cv2.drawContours(p,approx,-1,(255,0,0))

def item2():
    im1 = cv2.imread(image_folder+"Image1.jpg")
    im2 = cv2.imread(image_folder+"Image2.jpg")

    im1Gray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    cornersIm1 = cv2.goodFeaturesToTrack(im1Gray,4,0.01,10)
    cornersIm2 = cv2.goodFeaturesToTrack(im2Gray,4,0.01,10)

    detector = cv2.GFTTDetector_create(maxCorners=4,qualityLevel=0.01,minDistance=10)
    keypoints = detector.detect(im1Gray)

    print(type(keypoints[0]))

    #cv2.drawKeypoints(im1,keypoints,im1)
    #cv2.imshow("Imagem",im1)
    #cv2.waitKey()


if __name__ == '__main__':
    item2()