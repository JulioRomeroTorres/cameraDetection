import numpy as np
import cv2
from multiprocessing import Process

def webcam_video(): 

  cap = cv2.VideoCapture('C:/Users/julit/Downloads/Camara/video_11.mp4')
  while(True):
    ret, frame = cap.read()
    if ret == True:
      cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  cap.release()
  cv2.destroyAllWindows()

def local_video(): 
  path = 'C:/Users/julit/Downloads/Camara/video_4.mp4'
  cap = cv2.VideoCapture(path)
  while(True):
    ret, frame = cap.read()
    if ret == True:
     cv2.imshow('frame_2',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
  cap.release()
  cv2.destroyAllWindows()


def testProces(path,cameraID, id):
    print(cabeza)
    print('It is my id', str(id))
    cap = cv2.VideoCapture(path)
    while(True):
        ret, frame = cap.read()
        if ret == True:
            cv2.imshow('frame '+ str(id),frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()  
  

if __name__ == '__main__':
    #p1= Process(target = local_video)
    #p2= Process(target = webcam_video)

    pathVideos = 'C:/Users/julit/Downloads/Camara/video_'
    cabeza = 15
    p1= Process(target = testProces, args=(pathVideos + '4.mp4',4,1,))
    p2= Process(target = testProces, args=(pathVideos + '11.mp4',5,2,))

    p1.start() 
    p2.start()

    p1.join()
    p2.join()