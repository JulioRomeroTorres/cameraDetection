
from trafficCam import trafficCamera
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch

if __name__ == '__main__':
  
  pathVideos = 'C:/Users/julit/Downloads/Camara/video_'
  pathDataTrain = 'C:/Users/julit/Proyectos/cameraAA/dataset/train/'
  pathDataValid = 'C:/Users/julit/Proyectos/cameraAA/dataset/validation/'
  
  labelUsed = [ 2, 3, 5, 7 ]
  limitReg1 = [  (624,300), (310,8) ]
  limitReg2 = [ (606, 354), (376, 91), (378, 129), (341, 164), (228, 186), (141, 184), (153, 294), (192, 351), (590, 352) ]
  cameraD1  = trafficCamera(labelUsed, limitReg1, limitReg2)
  print('Camera 1: ',cameraD1.limitReg1, ' ', cameraD1.limitReg2)
  
  limitReg1 = [  (349,8), (322,8) ]
  limitReg2 = [ (587,346), (347,11), (321,10), (273,344)  ]
  cameraD2  = trafficCamera(labelUsed, limitReg1, limitReg2)
  print('Camera 2: ',cameraD2.limitReg1, ' ', cameraD2.limitReg2)
  
  limitReg1 = [  (371,12), (339,9) ]
  limitReg2 = [ (567,352), (375,18), (338,15), (268,348) ]
  cameraD3  = trafficCamera(labelUsed, limitReg1, limitReg2)
  print('Camera 3: ',cameraD3.limitReg1, ' ', cameraD3.limitReg2)
  
  #cameraD1.createDatset(2, 1, pathVideos, pathDataTrain)

  model =   torch.hub.load('ultralytics/yolov5','yolov5s')
  #model =   torch.hub.load('C:\\Users\\julit\Proyectos\\cameraAA\\yolov5','yolov5s')
  #model = torch.hub.load('.../yolov5','custom', path = 'models/last.pt')
  
  dataCamera1 = cv2.VideoCapture(pathVideos+'11.mp4')
  dataCamera2 = cv2.VideoCapture(pathVideos+'4.mp4')
  dataCamera3 = cv2.VideoCapture(pathVideos+'15.avi')

  cameraD1.displayVideo(dataCamera1, model)
  #cameraD2.displayVideo(dataCamera2, model)
  #cameraD3.displayVideo(dataCamera3, model)
  #dataCamera1.isOpened() or dataCamera3.isOpened() or dataCamera2.isOpened()

  '''while( True ):

    ret1, frame   = dataCamera1.read()
    frameDetect   = model(frame)
    cameraD1.getCenter(frameDetect, labelUsed)
    framemodDetect  = np.squeeze(frameDetect.render())
    framemodCir1 = cameraD1.drawCenter(framemodDetect)

    ret2, frame  = dataCamera2.read()
    frameDetect  = model(frame)
    cameraD2.getCenter(frameDetect, labelUsed)
    framemodDetect  = np.squeeze(frameDetect.render())
    framemodCir2 = cameraD2.drawCenter(framemodDetect)

    ret3, frame   = dataCamera3.read()
    frameDetect   = model(frame)
    cameraD3.getCenter(frameDetect, labelUsed)
    framemodDetect  = np.squeeze(frameDetect.render())
    framemodCir3 = cameraD3.drawCenter(framemodDetect)

    if ret1:
      cv2.imshow('Camero 1',framemodCir1)
    
    if ret2:
      cv2.imshow('Camero 2',framemodCir2)
    
    if ret3:
      cv2.imshow('Camero 3',framemodCir3)

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
  cv2.destroyAllWindows()'''

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


