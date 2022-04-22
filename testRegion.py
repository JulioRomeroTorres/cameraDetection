

import cv2
from cv2 import imshow

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
    count = 0

    while(video.isOpened()):
        ret, frame = video.read()
        print('Print dimension: ', frame.shape)
        cv2.imshow('frame', frame)

        if cv2.waitKey(10) == ord('s'):
          count = count + 1
          cv2.imwrite('C:/Users/julit/Proyectos/cameraAA/cameraDetection/dataset/train/' + str(count) + '.jpg' , frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()

def displayResul():

    frame = cv2.imread('C:/Users/julit/Proyectos/cameraAA/cameraDetection/dataset/train/1.jpg')
    #malFunc()

    print('Dimension: ', frame.shape)
    cv2.imshow("Frame With Detection",frame)
    cv2.setMouseCallback("Frame With Detection", getPoints )

    cv2.waitKey(0) 
        
    cv2.destroyAllWindows()

def displayVideo():

    image = cv2.imread('C:/Users/julit/Proyectos/cameraAA/cameraDetection/dataset/train/1.jpg')
    print('Print dim: ', image.shape)
    cv2.imshow('ddd',image)
    cv2.waitKey(0)
    print(labelUsed)

if __name__ == '__main__':

    labelUsed = [ 2, 3, 5, 7 ]
    #getLimitvideo()
    displayResul()
    



