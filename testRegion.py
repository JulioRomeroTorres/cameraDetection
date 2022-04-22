

import cv2
from cv2 import imshow

def malFunc():
    a = (1,2)
    b = (3,4)
    c = (2,6)
    d = (2,-4)

    ga = ((b[0] - a[0])*(c[1] - a[1]) - (b[1] - a[1])*(c[0] - a[0]))
    ge = ((b[0] - a[0])*(d[1] - a[1]) - (b[1] - a[1])*(d[0] - a[0]))

    image1 = cv2.imread('C:/Users/julit/Proyectos/cameraAA/test1.png')

    x = 300
    y = 400

    image2 = cv2.circle(image1, (x,y), radius = 10, color=(0, 0, 255), thickness=-1)

    cv2.imshow('rara', image2)

    cv2.waitKey(0) 
    
    #closing all open windows 
    cv2.destroyAllWindows() 


def displayVideo():
    global  labelUsed
    print(labelUsed)

if __name__ == '__main__':

    labelUsed = [ 2, 3, 5, 7 ]
    displayVideo()



