

import cv2
from cv2 import imshow
from mathFuncs import resizeImg
from mathFuncs import rezImg2
import numpy as np

def intoRegion(globalPos, limitReg ):

    dimLimit = len(limitReg)

    if( dimLimit == 2 ):
        into1Point = ((limitReg[1][0] - limitReg[0][0])*(globalPos[1] - limitReg[0][1]) - (limitReg[1][1] - limitReg[0][1])*(globalPos[0] - limitReg[0][0])) < 0  
        return into1Point

    elif( dimLimit >= 3 ):
        auxArr = []
        for i in range(0,dimLimit):
            into1Point = ((limitReg[(i+1)%dimLimit][0] - limitReg[i][0])*(globalPos[1] - limitReg[i][1]) - (limitReg[(i+1)%dimLimit][1] - limitReg[i][1])*(globalPos[0] - limitReg[i][0])) < 0 
            auxArr.append(int(into1Point))
            print('----------')
            print( 'Limites: ', limitReg[i][0] , limitReg[i][1], limitReg[(i+1)%dimLimit][0], limitReg[(i+1)%dimLimit][1] )
            print('Is valid: ', into1Point)
        
        if( sum(auxArr) == dimLimit ):
            return True
        else:
            return False

def malFunc():
    a = (1515,605)
    b = (810,17)
    c = (1200,470)
    d = (2,-4)
    
    ga = ((b[0] - a[0])*(c[1] - a[1]) - (b[1] - a[1])*(c[0] - a[0]))
    
    a = (1515,605)
    b = (810,17)
    c = (1457,331)

    ge = ((b[0] - a[0])*(c[1] - a[1]) - (b[1] - a[1])*(c[0] - a[0]))
    
    print('Raa: ', ga, ' ge: ',ge )

    image1 = cv2.imread('C:/Users/julit/Downloads/Camara/truckTest5.png')

    x = 300
    y = 400

    image2 = cv2.circle(image1, c, radius = 10, color=(0, 0, 255), thickness=-1)

    cv2.imshow('rara', image2)

    cv2.waitKey(0) 
    
    #closing all open windows 
    cv2.destroyAllWindows() 

def getPoints(event, x, y, flag, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x,y)
    
def getLimitvideo():

    video = cv2.VideoCapture('C:/Users/julit/Downloads/Camara/video_11.mp4')
    #video = cv2.VideoCapture('rtsp://admin:dcsautomation123@192.168.0.102/80')
    count = 0

    while(video.isOpened()):
        ret, frame = video.read()
        frame = resizeImg(frame,100)
        print('Print dimension: ', frame.shape)
        cv2.imshow('frame', frame)

        if cv2.waitKey(10) == ord('s'):
          count = count + 1
          cv2.imwrite('C:/Users/julit/Proyectos/cameraAA/cameraDetection/limits/' + str(count) + '.jpg' , frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()

def displayResul():

    frame = cv2.imread('C:/Users/julit/Proyectos/cameraAA/cameraDetection/limits/camera_15.jpg')
    #malFunc()
    #frame = rezImg2(frame,(320,320))
    print('Dimension: ', frame.shape)
    cv2.imshow("Frame With Detection",frame)
    cv2.setMouseCallback("Frame With Detection", getPoints )

    cv2.waitKey(0) 

        
    cv2.destroyAllWindows()

def displayVideo():

    
    image = cv2.imread('C:/Users/julit/Proyectos/cameraAA/cameraDetection/dataset/train/1.jpg')
    #image = cv2.imread('C:/Users/julit/Proyectos/cameraAA/cameraDetection/dataset/train/1.jpg')
    #print('Print dim: ', image.shape)
    cv2.imshow('ddd',image)
    cv2.waitKey(0)
    print(labelUsed)

if __name__ == '__main__':

    labelUsed = [ 2, 3, 5, 7 ]
    limitReg  = np.array([ [481, 200], [415, 127], [366, 81], [378, 119], [366, 149], [269, 180], [195, 180], [190, 276], [332, 264], [409, 243], [450, 208] ])
    #print( intoRegion(  (400, 197), limitReg) )
    #getLimitvideo()
    displayResul()
    







