from concurrent.futures import ThreadPoolExecutor
import threading
import time
import cv2

def aeaMongol(dataCamera, camaraD, id):
    ret, frame   = dataCamera.read()
    print('ret ', ret)
    if ret:
        print('It is possible: ' + str(id) )
        print('teh frame: ', frame)
        cv2.imshow('Camera ' + str(id), frame) 
    time.sleep(0.03)
    print('Aea Mogol sumate esta: ')

if __name__ == '__main__':

    pathVideos = 'C:/Users/julit/Downloads/Camara/video_'
    dataCamera1 = cv2.VideoCapture(pathVideos + '4.mp4')
    dataCamera2 = cv2.VideoCapture(pathVideos + '11.mp4')
    dataCamera3 = cv2.VideoCapture(pathVideos + '15.avi')
    dataCam     = cv2.VideoCapture(pathVideos + '16.avi ')  
    
    cameraD1 = 0.1
    cameraD2 = 0.1
    cameraD3 = 0.1

    arrThread = []

    while (True):
        
        ret, frame   = dataCam.read()
        if ret:
            print('It is possible: ' )
            cv2.imshow('Camera ', frame) 

        executorCam = ThreadPoolExecutor(max_workers = 3)
        executorCam.submit(aeaMongol, dataCamera1, cameraD1, 1)
        executorCam.submit(aeaMongol, dataCamera2, cameraD2, 2)
        executorCam.submit(aeaMongol, dataCamera3, cameraD3, 3)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
          
    dataCamera1.release()
    dataCamera2.release()
    dataCamera3.release()
    dataCam.release()
    cv2.destroyAllWindows()