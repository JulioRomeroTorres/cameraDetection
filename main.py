
from trafficCam import trafficCamera
from controller import plcS7
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch

if __name__ == '__main__':
  
  pathVideos = 'C:/Users/julit/Downloads/Camara/video_'

  refCamera1 = (142 ,188)
  refCamera2 = (348 ,355)
  refCamera3 = (353, 352)

  # Change to real scale
  scaleCamera1 = 0.08
  scaleCamera2 = 0.08
  scaleCamera3 = 0.08
  
  labelUsed = [ 0, 1 ]
  limitReg1 = [ (539 ,233), (380 ,82) ]
  limitReg2 = [ (481 ,200), (366 ,84), (378 ,105), (376 ,137), (337 ,164), (281 ,180), (225 ,184), (170 ,182), (154 ,265), (226 ,269), (322 ,262), (393 ,243) , (476 ,194) ]
  cameraD1  = trafficCamera( scaleCamera1, refCamera1, labelUsed, limitReg1, limitReg2)
  
  limitReg1 = [  (349,8), (322,8) ]
  limitReg2 = [ (587,346), (347,11), (321,10),(273,344)  ]
  cameraD2  = trafficCamera( scaleCamera2, refCamera2, labelUsed, limitReg1, limitReg2)
  
  limitReg1 = [  (371,12), (339,9) ]
  limitReg2 = [ (567,352), (375,18), (338,15), (268,348) ]
  cameraD3  = trafficCamera( scaleCamera3, refCamera3, labelUsed, limitReg1, limitReg2)
  
  ipPlc   = '192.168.252.12'
  rackPlc = 0
  slotPlc = 1
  plcSem =   plcS7(ipPlc, rackPlc, slotPlc)

  model = torch.hub.load('../yolov5','custom', path = 'models/best.pt', source = 'local')
  
  dataCamera1 = cv2.VideoCapture(pathVideos+'11.mp4')
  dataCamera2 = cv2.VideoCapture(pathVideos+'4.mp4')
  dataCamera3 = cv2.VideoCapture(pathVideos+'15.avi')

  while( True ):

    ret1, frame   = dataCamera1.read()
    frameDetect   = model(frame)
    cameraD1.getCenter(frameDetect, labelUsed)
    framemodDetect  = np.squeeze(frameDetect.render())
    framemodCir1, arrCar, arrTruck = cameraD1.drawCenter(framemodDetect)
    #plcSem.sendData(np.array(arrTruck).sum())

    ret2, frame  = dataCamera2.read()
    frameDetect  = model(frame)
    cameraD2.getCenter(frameDetect, labelUsed)
    framemodDetect  = np.squeeze(frameDetect.render())
    framemodCir2, arrCar, arrTruck = cameraD2.drawCenter(framemodDetect)
    #plcSem.sendData(np.array(arrCar).sum())

    ret3, frame   = dataCamera3.read()
    frameDetect   = model(frame)
    cameraD3.getCenter(frameDetect, labelUsed)
    framemodDetect  = np.squeeze(frameDetect.render())
    framemodCir3, arrCar, arrTruck = cameraD3.drawCenter(framemodDetect)
    #plcSem.sendData(np.array(arrTruck).sum())
    
    if ret1:
      cv2.imshow('Camero 1', framemodCir1)
    
    if ret2:
      cv2.imshow('Camero 2', framemodCir2)
    
    if ret3:
      cv2.imshow('Camero 3', framemodCir3)

    cameraD1.carLabel.destroyObject()
    cameraD1.motorLabel.destroyObject()
    cameraD1.busLabel.destroyObject()
    cameraD1.truckLabel.destroyObject()

    cameraD2.carLabel.destroyObject()
    cameraD2.motorLabel.destroyObject()
    cameraD2.busLabel.destroyObject()
    cameraD2.truckLabel.destroyObject()

    cameraD3.carLabel.destroyObject()
    cameraD3.motorLabel.destroyObject()
    cameraD3.busLabel.destroyObject()
    cameraD3.truckLabel.destroyObject()

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
          
  dataCamera1.release()
  dataCamera2.release()
  dataCamera3.release()
  cv2.destroyAllWindows()



