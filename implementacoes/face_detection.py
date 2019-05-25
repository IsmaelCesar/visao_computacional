"""
Implementation of the algorithm:
CHEN, Yao-Jiunn; LIN, Yen-Chun. Simple face-detection algorithm based on minimum facial features.
In: IECON 2007-33rd Annual Conference of the IEEE Industrial Electronics Society. IEEE, 2007. p. 455-460.
"""
import cv2

global images_folder

images_folder = 'imagens/'


def skin_detection(image,F1,F2,White):
    """
    :param image: Image to be processed, It is a convention of the openCV library
    that the images come in the BGR(Blue, Green and Red) format
    :param F1:    Function 1 for the red color
    :param F2:    Function 2 for the green color
    :param White: Function for detecting the white range of colors in an image
    :return: image with the skin features
    """
    epsilon = .000000001
    mImage = image.copy()
    mImage = mImage/255
    for i in range(mImage.shape[0]):
        for j in range(mImage.shape[1]):
            r = mImage[i,j,2]/(mImage[i,j,0]+mImage[i,j,1]+mImage[i,j,2]+epsilon)
            g = mImage[i, j, 1]/(mImage[i,j,0]+mImage[i,j,1]+mImage[i,j,2]+epsilon)
            r1 = F1(r)
            r2 = F2(r)
            w  = White(r,g)
            if(not(g < r1  and g > r2 and w > 0.001 )):
                mImage [i,j,0] = 0.
                mImage [i,j,1] = 0.
                mImage [i,j,2] = 0.
    #mImage = mImage*255
    return mImage

def  main():
    image = cv2.imread(images_folder+'Lenna.png')

    F1 =  lambda eta : -1.37*eta**2 + 1.0743*eta + 0.2
    F2 =  lambda epsilon : -0.776*epsilon**2 + 0.5601*epsilon + 0.18
    White = lambda r,g: (r - 0.33)**2 + (g - 0.33)**2
    newImage = skin_detection(image,F1,F2,White)

    cv2.imshow("Original",image)
    cv2.imshow("Skin Detection", newImage)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()