
from trafficCam import trafficCamera
from controller import plcS7
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch

if __name__ == '__main__':
  
  pathVideos = 'C:/Users/julit/Downloads/Camara/video_'
  pathDataTrain = 'C:/Users/julit/Proyectos/cameraAA/dataset/train/'
  pathDataValid = 'C:/Users/julit/Proyectos/cameraAA/dataset/validation/'

  refCamera1 = (0.0,0.0)
  refCamera2 = (0.0,0.0)
  refCamera3 = (0.0,0.0)

  scaleCamera1 = 15.0
  scaleCamera2 = 15.0
  scaleCamera3 = 15.0
  
  labelUsed = [ 0, 1 ]
  limitReg1 = [  (624,300), (310,8) ]
  limitReg2 = [ (606, 354), (376, 91), (378, 129), (341, 164), (228, 186), (141, 184), (153, 294), (192, 351), (590, 352) ]
  cameraD1  = trafficCamera( scaleCamera1, refCamera1, labelUsed, limitReg1, limitReg2)
  
  limitReg1 = [  (349,8), (322,8) ]
  limitReg2 = [ (587,346), (347,11), (321,10),(273,344)  ]
  cameraD2  = trafficCamera( scaleCamera2, refCamera2, labelUsed, limitReg1, limitReg2)
  
  limitReg1 = [  (371,12), (339,9) ]
  limitReg2 = [ (567,352), (375,18), (338,15), (268,348) ]
  cameraD3  = trafficCamera( scaleCamera3, refCamera3, labelUsed, limitReg1, limitReg2)
  
  ipPlc   = '192.168.0.1'
  rackPlc = 0
  slotPlc = 1
  plc1 =   plcS7(ipPlc, rackPlc, slotPlc)

  ipPlc   = '192.168.0.1'
  rackPlc = 0
  slotPlc = 1
  plc2 =   plcS7(ipPlc, rackPlc, slotPlc)
  
  ipPlc   = '192.168.0.1'
  rackPlc = 0
  slotPlc = 1
  plc3 =   plcS7(ipPlc, rackPlc, slotPlc)

  #cameraD1.createDatset(2, 1, pathVideos, pathDataTrain)

  #model =   torch.hub.load('ultralytics/yolov5','yolov5s')
  #model = torch.hub.load('ultralytics/yolov5','custom', path = 'models/best.pt')
  model = torch.hub.load('../yolov5','custom', path = 'models/best.pt', source = 'local')
  
  dataCamera1 = cv2.VideoCapture(pathVideos+'11.mp4')
  dataCamera2 = cv2.VideoCapture(pathVideos+'4.mp4')
  dataCamera3 = cv2.VideoCapture(pathVideos+'15.avi')

  #cameraD1.displayVideo(dataCamera1, model)
  #cameraD2.displayVideo(dataCamera2, model)
  #cameraD3.displayVideo(dataCamera3, model)
  #dataCamera1.isOpened() or dataCamera3.isOpened() or dataCamera2.isOpened()

  while( True ):

    ret1, frame   = dataCamera1.read()
    frameDetect   = model(frame)
    cameraD1.getCenter(frameDetect, labelUsed)
    framemodDetect  = np.squeeze(frameDetect.render())
    framemodCir1, arrCar, arrTruck = cameraD1.drawCenter(framemodDetect)
    #plc1.sendData(arrCar)
    #framemodCT1 = cameraD1.putDistance(framemodCir1)

    ret2, frame  = dataCamera2.read()
    frameDetect  = model(frame)
    cameraD2.getCenter(frameDetect, labelUsed)
    framemodDetect  = np.squeeze(frameDetect.render())
    framemodCir2, arrCar, arrTruck = cameraD2.drawCenter(framemodDetect)
    #plc2.sendData(arrTruck)
    #framemodCT2 = cameraD2.putDistance(framemodCir2)

    ret3, frame   = dataCamera3.read()
    frameDetect   = model(frame)
    cameraD3.getCenter(frameDetect, labelUsed)
    framemodDetect  = np.squeeze(frameDetect.render())
    framemodCir3, arrCar, arrTruck = cameraD3.drawCenter(framemodDetect)
    #plc3.sendData(arrTruck)
    
    #framemodCT3 = cameraD3.putDistance(framemodCir3)

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

  '''frame = cv2.imread('C:/Users/julit/Downloads/Camara/truckTest4.png')
  frameDetect   = model(frame)
  getCenter(frameDetect, labelUsed)
  framemodDetect  = np.squeeze(frameDetect.render())
  print('size fraame: ', frame.shape, ' modified size: ', framemodDetect.shape )
  plt.imshow(framemodDetect)
  plt.show()

  framemodCir = drawCenter(framemodDetect)

  carLabel.destroyObject()
  motorLabel.destroyObject()
  busLabel.destroyObject()
  truckLabel.destroyObject()

  cv2.imshow('Frame With Detection',framemodCir)
  cv2.waitKey(0)
    
  cv2.destroyAllWindows()'''


