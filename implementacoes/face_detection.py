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
                 'person4.jpg',
                 'person5.jpg',
                 'person6.jpg',
                 'person7.jpg',
                 'persons.jpg']

global width, height

def hair_detection(image,H,I,ArcCoss,NormRGB):
    global rows, cols
    kImage = image.copy()
    mImage = kImage.copy()/255
    mImage = np.array([[NormRGB(mImage[i][j][2], mImage[i][j][1], mImage[i][j][0]) for j in range(cols)] for i in range(rows)])
    rImage = mImage[:, :, 2]
    gImage = mImage[:, :, 1]
    bImage = mImage[:, :, 0]


    angles = ArcCoss(rImage,gImage,bImage)
    applyH = np.array([[H(bImage[i][j],gImage[i][j],angles[i][j]) for j in range(cols)] for i in range(rows)])
    applyI = I(rImage,gImage,bImage)
    #applyI = I(kImage[:,:,2], kImage[:,:,1], kImage[:,:,0])
    # Computing B - G and B - R
    bmg = bImage - gImage
    bmr = bImage - rImage
    #bmr = kImage[:, :, 0] - kImage[:, :, 2]
    #bmg = kImage[:,:,0] - kImage[:,:,1]

    for i in range(rows):
        for j in range(cols):
            condition = (applyI[i,j] < .31372549)#80/255 ~ 0.31372549
            condition = condition and (bmg[i,j] < .058823529 or bmr[i,j] < .058823529) #  15/255 ~ 0.058823529
            condition = condition or (applyH[i,j] > np.pi/4 and applyH[i,j] <= 1/2*np.pi )# np.pi/3  2/9*np.pi
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


def skin_quantization(skin_detect):
    global  rows,cols
    gray_skin_detect = cv2.cvtColor(skin_detect,cv2.COLOR_BGR2GRAY)
    _,binary_skin = cv2.threshold(gray_skin_detect,0,255,cv2.THRESH_BINARY)
    kernel = np.ones((10,10),dtype=np.uint8)
    quantization = cv2.erode(binary_skin,kernel,iterations=2)
    quantization = cv2.dilate(quantization, kernel, iterations=2)
    return quantization

def hair_quantization(hair_detect):
    global  rows,cols
    gray_hair_detect = cv2.cvtColor(hair_detect,cv2.COLOR_BGR2GRAY)
    _,binary_hair = cv2.threshold(gray_hair_detect,0,255,cv2.THRESH_BINARY)
    kernel = np.ones((3,3),dtype=np.uint8)
    quantization = cv2.erode(binary_hair,kernel,iterations=1)
    quantization = cv2.dilate(quantization, kernel, iterations=1)
    return quantization

def size_filter_single_face(labels,stats,centroids,xy_positions):
    global rows,cols

    greater = 0
    component_label = 0
    component_centroid = None
    component_stats = None
    component_positions= None
    for i,s in enumerate(stats):
        el = s[-1]
        if el > greater :
            greater=el
            component_label = i
            component_centroid = centroids[i]
            component_stats = stats[i]
            component_positions = np.array([xy_positions[i]])

    for i in  range(rows):
        for j in range(cols):
            if labels[i,j] != component_label:
                labels[i, j] = 0

    return  labels, component_stats, component_centroid,component_positions

def size_filter_multi_face(labels,stats,centroids,xy_positions):
    #For multiface detection, any component larger tha 50 pixels shall be preserved
    global rows,cols
    greater = 50
    component_label = []
    component_centroid = []
    component_stats = []
    component_positions= []
    for i,s in enumerate(stats):
        el = s[-1]
        if el > greater :
            component_label.append(i)
            component_centroid.append(centroids[i])
            component_stats.append(stats[i])
            component_positions.append(np.array(xy_positions[i]))

    for l in component_label:
        for i in  range(rows):
            for j in range(cols):
                if labels[i,j] != l:
                    labels[i, j] = 0

    return  labels, component_stats, component_centroid,component_positions


def size_filter(skinLabels,skinStats,skinCentroids,s_xy_positions,is_single_face=True):
    # appling size filter
    if is_single_face:
        if (len(skinStats) > 1):
            skinLabels, skinStats, skinCentroids, s_xy_positions = size_filter_single_face(skinLabels, skinStats,
                                                                                        skinCentroids,s_xy_positions)
    else:
        skinLabels,skinStats,skinCentroids,s_xy_positions = size_filter_multi_face(skinLabels,skinStats,
                                                                                   skinCentroids,s_xy_positions)

    return skinLabels, skinStats, skinCentroids, s_xy_positions

def compute_boudingbox_from_skin_hair_component_features(skinStats,hairStats):
    s_xy_positions = []
    h_xy_positions = []
    # Compute x y position
    for s in skinStats:
        #Bouding box of skin components
        s_x_min, s_y_min = s[0], s[1]
        s_x_max = s[0] + s[2]
        s_y_max = s[1] + s[3]

        # The vertices of the bounding box are being computed in a clockwise manner
        s_xy_positions.append([[s_x_min, s_y_min], [s_x_min, s_y_max], [s_x_max, s_y_max], [s_x_max, s_y_min]])

    s_xy_positions = np.array(s_xy_positions)

    for h in hairStats:
        # Bounding box hair
        h_x_min, h_y_min = h[0], h[1]
        h_x_max = h[0] + h[2]
        h_y_max = h[1] + h[3]
        h_xy_positions.append([[h_x_min, h_y_min], [h_x_min, h_y_max], [h_x_max, h_y_max], [h_x_max, h_y_min]])


    h_xy_positions = np.array(h_xy_positions)

    return s_xy_positions, h_xy_positions

def compute_skin_hair_component_labeling(qSkin,qHair):
    skinNumLabels, skinLabels, skinStats, skinCentroids = cv2.connectedComponentsWithStats(qSkin)
    hairNumLabels, hairLabels, hairStats, hairCentroids = cv2.connectedComponentsWithStats(qHair)

    # ignoring background label stasts and centroids and bounding box positons
    skinStats = skinStats[1:]
    hairStats = hairStats[1:]

    skinCentroids = skinCentroids[1:]
    hairCentroids = hairCentroids[1:]

    return skinLabels,skinStats,skinCentroids,hairLabels,hairStats,hairCentroids

def compute_boxes_intercection(s_xy_positions,h_xy_positions):
    hair_boxes = []
    skin_hair_intersect = []
    intesect_area = 0
    for h_xy in h_xy_positions:
        for s_xy in s_xy_positions:
            xI_0 = max(h_xy[0][0], s_xy[0][0])
            yI_0 = max(h_xy[0][1], s_xy[0][1])

            xI_1 = min(h_xy[1][0],s_xy[1][0])
            yI_1 = max(h_xy[1][1], s_xy[1][1])

            xI_2 = min(h_xy[2][0], s_xy[2][0])
            yI_2 = min(h_xy[2][1], s_xy[2][1])

            xI_3 = max(h_xy[3][0], s_xy[3][0])
            yI_3 = min(h_xy[3][1], s_xy[3][1])

            area = max(0,xI_0 - xI_0 +1) *max(0,yI_2 - yI_2 + 1)
            if area > intesect_area:
                hair_boxes.append(h_xy)
                #skin_hair_intersect.append([[xI_0,yI_0],[xI_1,yI_1],[xI_2,yI_2],[xI_3,yI_3]])
                skin_hair_intersect.append(s_xy)

    return np.array(hair_boxes),np.array(skin_hair_intersect)

def detect_image(image,bouding_boxes):
    detect = image.copy()
    for i, xy in enumerate(bouding_boxes):
        cv2.rectangle(detect, (xy[0, 0], xy[0, 1]), (xy[2, 0], xy[2, 1]), color=(0, 0, 255), thickness=3)

    return detect

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


def detect_single_face():
    global rows, cols
    image = cv2.imread(images_folder + image_selected[0])
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

    qSkin = skin_quantization(skinDetect)
    qHair = hair_quantization(hairDetect)


    skinLabels,skinStats,skinCentroids,hairLabels,hairStats,hairCentroids = compute_skin_hair_component_labeling(qSkin,qHair)

    s_xy_pos, h_xy_pos = compute_boudingbox_from_skin_hair_component_features(skinStats,hairStats)

    skinLabels,skinStats,skinCentroids,s_xy_pos = size_filter(skinLabels,skinStats,skinCentroids,s_xy_pos)

    hair_box,intersect = compute_boxes_intercection(s_xy_pos,h_xy_pos)

    detect = detect_image(image, intersect)

    cv2.imshow("Original",image)
    cv2.imshow("Detection",detect)
    cv2.waitKey(0)


def detect_multiface():
    global rows, cols
    image = cv2.imread(images_folder + image_selected[7])
    rows = image.shape[0]
    cols = image.shape[1]

    F1 = getF1()
    F2 = getF2()
    H = getH()
    ArcCos = getArccosFunction()
    NormRGB = getNormRGB()
    I = getI()
    White = whiteRange()

    skinDetect = skin_detection(image, F1, F2, White, H, ArcCos, NormRGB)

    hairDetect = hair_detection(image, H, I, ArcCos, NormRGB)

    qSkin = skin_quantization(skinDetect)
    qHair = hair_quantization(hairDetect)



    skinLabels, skinStats, skinCentroids, hairLabels, hairStats, hairCentroids = compute_skin_hair_component_labeling(
        qSkin, qHair)

    s_xy_pos, h_xy_pos = compute_boudingbox_from_skin_hair_component_features(skinStats, hairStats)

    skinLabels, skinStats, skinCentroids, s_xy_pos = size_filter(skinLabels, skinStats, skinCentroids, s_xy_pos,
                                                                 is_single_face=False)

    hair_box, intersect = compute_boxes_intercection(s_xy_pos, h_xy_pos)

    detect = detect_image(image, intersect)

    cv2.imshow("Original", image)
    cv2.imshow("Detection", detect)
    cv2.waitKey(0)


if __name__ == '__main__':
    detect_single_face()
    #detect_multiface()