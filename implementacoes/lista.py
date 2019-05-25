import  cv2
import  numpy as np

global image_folder
image_folder = "imagens/"



def drawCirclesAndShowImage(image,corners):
    dstCirclesIm = image.copy()
    for c in corners:
        dstCirclesIm = cv2.circle(dstCirclesIm, (c[0][0], c[0][1]), 10, (0, 0, 255))

    cv2.imshow("Circles1",dstCirclesIm)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def keyPoints2NumpyArray(keypoints):

    newKp = []
    for kp in keypoints:
        newKp.append([kp.pt[0],kp.pt[1]])
    return np.array(newKp,dtype=np.float32)

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

    rows,cols = im1Gray.shape

    orb = cv2.ORB_create(nfeatures=500)
    im1Kp = orb.detect(im1Gray)
    im1Kp,im1Desc = orb.compute(im1Gray,im1Kp)

    im2Kp = orb.detect(im2Gray)
    im2Kp,im2Desc = orb.compute(im2Gray, im2Kp)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(im1Desc, im2Desc,2)

    good = []
    for m,n in matches:
        if m.distance/n.distance < 0.5:
            good.append([m])
    trainKp = []
    queryKp = []
    veryGood=[]
    for i  in range(len(good)):
        if len(trainKp) < 4:
            trainKp.append(im1Kp[good[i][0].trainIdx])
            queryKp.append(im2Kp[good[i][0].queryIdx])
            veryGood.append(good[i])
        else:
            for j in range(4):
                if good[i][0].distance/veryGood[j][0].distance < 0.75:
                    trainKp[j] = im1Kp[good[i][0].trainIdx]
                    queryKp[j] = im2Kp[good[i][0].queryIdx]
                    veryGood[j] = good[i]
                    break

    #newKpIm1 = cv2.goodFeaturesToTrack(im1Gray,maxCorners=4,qualityLevel=0.01,minDistance=10)
    #newKpIm2 = cv2.goodFeaturesToTrack(im2Gray,maxCorners=4,qualityLevel=0.01,minDistance=10)
    """
    img3 = np.zeros((rows, cols, 3))
    img3 = cv2.drawMatchesKnn(im1, trainKp, im2, queryKp, veryGood, img3)
    cv2.imshow("Matches Found", img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    #M, mask = cv2.findHomography(newKpIm1, newKpIm2, method=cv2.RANSAC)
    newKpIm1 = keyPoints2NumpyArray(trainKp)
    newKpIm2 = keyPoints2NumpyArray(queryKp)

    M = cv2.getPerspectiveTransform(newKpIm1,newKpIm2)

    dst = np.zeros((rows,cols+600,3))
    dst = cv2.warpPerspective(im2,M,(cols+600,rows))
    dst[0:rows, 0:cols] = im1
    cv2.imshow("Result",dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def item2_2():
    im1 = cv2.imread(image_folder + "Image1.jpg")
    im2 = cv2.imread(image_folder + "Image2.jpg")

    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    rows, cols = im1Gray.shape

    orb = cv2.ORB_create(nfeatures=500)
    im1Kp = orb.detect(im1Gray)
    im1Kp, im1Desc = orb.compute(im1Gray, im1Kp)

    im2Kp = orb.detect(im2Gray)
    im2Kp, im2Desc = orb.compute(im2Gray, im2Kp)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(im1Desc, im2Desc, 2)

    good = []
    for m, n in matches:
        if m.distance / n.distance < 0.5:
            good.append([m])
    trainKp = []
    queryKp = []
    veryGood = []
    for i in range(len(good)):
        if len(trainKp) < 4:
            trainKp.append(im1Kp[good[i][0].trainIdx])
            queryKp.append(im2Kp[good[i][0].queryIdx])
            veryGood.append(good[i])
        else:
            for j in range(4):
                if good[i][0].distance / veryGood[j][0].distance <= 0.65:
                    trainKp[j] = im1Kp[good[i][0].trainIdx]
                    queryKp[j] = im2Kp[good[i][0].queryIdx]
                    veryGood[j] = good[i]
                    break


    trainKp = keyPoints2NumpyArray(trainKp)
    queryKp = keyPoints2NumpyArray(queryKp)

    M, mask = cv2.findHomography(trainKp, queryKp, method=cv2.RANSAC)

    trainKp = trainKp.reshape(-1, 1, 2)
    queryKp = queryKp.reshape(-1, 1, 2)
    features2_ = cv2.perspectiveTransform(queryKp,M)

    pts = np.concatenate((trainKp,features2_),axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin, -ymin]
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])

    result = cv2.warpPerspective(im2,M, (cols+400, rows))
    #result[0:rows, 0 : cols] = im1

    cv2.imshow("result",result)
    cv2.waitKey(0)

if __name__ == '__main__':
    item1()
    #item2()
