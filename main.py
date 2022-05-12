
from trafficCam import trafficCamera
from controller import plcS7
from mathFuncs import resizeImg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch

if __name__ == '__main__':
  
  pathVideos = 'C:/Users/julit/Downloads/Camara/video_'
  deviceCuda = torch.device("cuda")

  refCamera1 = (142 ,188)
  refCamera2 = (352, 356)
  refCamera3 = (350, 352)

  # Change to real scale
  scaleCamera1 = 0.08
  scaleCamera2 = 0.08
  scaleCamera3 = 0.08
    
  labelUsed = [ 0, 1 ]
  limitReg1 = [(443, 351), (330, 6)]
  limitReg2 = [(443, 351), (330, 6), (318, 5), (262, 348)]
  cameraD1  = trafficCamera( scaleCamera1, refCamera1, labelUsed, limitReg1, limitReg2)
  
  limitReg1 = [(566, 267), (292, 18)]
  limitReg2 = [(481, 200), (415, 127), (366, 81), (378, 119), (366, 149), (269, 180), (195, 180), (190, 276), (332, 264), (409, 243), (450, 208)]
  cameraD2  = trafficCamera( scaleCamera2, refCamera2, labelUsed, limitReg1, limitReg2)
  
  limitReg1 = [(445, 324), (356, 12)]
  limitReg2 = [(432, 327), (350, 13), (338, 9), (268, 316) ]
  cameraD3  = trafficCamera( scaleCamera3, refCamera3, labelUsed, limitReg1, limitReg2)
  
  ipPlc   = '192.168.0.120'
  rackPlc = 0
  slotPlc = 1
  plcSem =   plcS7(ipPlc, rackPlc, slotPlc)

  model = torch.hub.load('../yolov5','custom', path = 'models/best.pt', source = 'local')
  model.to(deviceCuda)
  
  dataCamera1 = cv2.VideoCapture(pathVideos + '4.mp4')
  dataCamera2 = cv2.VideoCapture(pathVideos + '11.mp4')
  dataCamera3 = cv2.VideoCapture(pathVideos + '15.avi')

  '''dataCamera1 = cv2.VideoCapture('rtsp://admin:dcsautomation123@192.168.0.100/80')
  dataCamera2 = cv2.VideoCapture('rtsp://admin:dcsautomation123@192.168.0.101/80')
  dataCamera3 = cv2.VideoCapture('rtsp://admin:dcsautomation123@192.168.0.102/80')'''

  while( True ):

    ret1, frame   = dataCamera1.read()    
    if ret1:
      
      frame = resizeImg(frame,50)
      frameDetect   = model(frame)
      cameraD1.getCenter(frameDetect, labelUsed)
      framemodDetect  = np.squeeze(frameDetect.render())
      framemodCir1, arrCar, arrTruck = cameraD1.drawCenter(framemodDetect)
      #cv2.imshow('Camero 1 Ori', frame)
      cv2.imshow('Camero 1', framemodCir1)
      #plcSem.sendData(np.array(arrTruck).sum())

    ret2, frame  = dataCamera2.read()
    if ret2:

      frame = resizeImg(frame,50)
      frameDetect  = model(frame)
      cameraD2.getCenter(frameDetect, labelUsed)
      framemodDetect  = np.squeeze(frameDetect.render())
      framemodCir2, arrCar, arrTruck = cameraD2.drawCenter(framemodDetect)
      #plcSem.sendData(np.array(arrCar).sum())
      #cv2.imshow('Camero 2 Ori', frame)
      cv2.imshow('Camero 2', framemodCir2)
    
    ret3, frame   = dataCamera3.read()
    if ret3:

      frame = resizeImg(frame,50)
      frameDetect   = model(frame)
      cameraD3.getCenter(frameDetect, labelUsed)
      framemodDetect  = np.squeeze(frameDetect.render())
      framemodCir3, arrCar, arrTruck = cameraD3.drawCenter(framemodDetect)
      #plcSem.sendData(np.array(arrTruck).sum())
      #cv2.imshow('Camero 3 Ori', frame)
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



