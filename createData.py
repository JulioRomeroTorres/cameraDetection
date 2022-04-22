from ast import increment_lineno
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch

class labelDetected:
  def __init__(self):
    self.arrCentx   = []
    self.arrCenty   = []
    self.arrWeight  = []
    self.arrHeight  = []
    self.count      = 0
    self.arrPrecis  = []
    self.arrValid   = []
  
  def defineCenter(self, posx1, posy1, posx2, posy2 ):

    self.arrCentx.append(int( 0.5*(posx1 + posx2)))
    self.arrCenty.append(int( 0.5*(posy1 + posy2)))
    self.arrWeight.append(posx2 - posx1)
    self.arrWeight.append(posy2 - posy1)
  
  def destroyObject(self):
    self.arrCentx   = []
    self.arrCenty   = []
    self.arrWeight  = []
    self.arrHeight  = []
    self.count      = 0
    self.arrPrecis  = []
    self.arrValid   = []

class trafficCamera():

  def __init__(self, labelUsed, limitReg1, limitReg2):
    self.carLabel     = labelDetected()
    self.motorLabel   = labelDetected()
    self.busLabel     = labelDetected()
    self.truckLabel   = labelDetected()
    self.labelUsed    = labelUsed
    self.limitReg1    = limitReg1
    self.limitReg2    = limitReg2

  def getCenter(self, frameDetect, arrIdlabel):
    
    resultPred = frameDetect.xyxy
    tensorDim = list(resultPred[0].size())

    #print('Result Prediction: ', resultPred[0])
    for i in range(0, tensorDim[0] ):
      if resultPred[0][i,5].item() == arrIdlabel[0]:
        self.carLabel.defineCenter(resultPred[0][i,0].item(), resultPred[0][i,1].item(), resultPred[0][i,2].item(), resultPred[0][i,3].item() )  
        
        self.carLabel.count = self.carLabel.count + 1
        self.carLabel.arrPrecis.append(resultPred[0][i,4].item())

      elif resultPred[0][i,5].item() == arrIdlabel[1]: 
        self.motorLabel.defineCenter(resultPred[0][i,0].item(), resultPred[0][i,1].item(), resultPred[0][i,2].item(), resultPred[0][i,3].item() )
        self.motorLabel.count = self.motorLabel.count + 1
        self.motorLabel.arrPrecis.append(resultPred[0][i,4].item())

      elif resultPred[0][i,5].item() == arrIdlabel[2]:
        self.busLabel.defineCenter(resultPred[0][i,0].item(), resultPred[0][i,1].item(), resultPred[0][i,2].item(), resultPred[0][i,3].item() )
        self.busLabel.count = self.busLabel.count + 1
        self.busLabel.arrPrecis.append(resultPred[0][i,4].item())

      elif resultPred[0][i,5].item() == arrIdlabel[3]:
        self.truckLabel.defineCenter(resultPred[0][i,0].item(), resultPred[0][i,1].item(), resultPred[0][i,2].item(), resultPred[0][i,3].item() )  
        self.truckLabel.count = self.truckLabel.count + 1
        self.truckLabel.arrPrecis.append(resultPred[0][i,4].item())
      
  def intoRegion(self, globalPos, limitReg ):

    dimLimit = len(limitReg)
    
    if( dimLimit == 2 ):
      into1Point = ((limitReg[1][0] - limitReg[0][0])*(globalPos[1] - limitReg[0][1]) - (limitReg[1][1] - limitReg[0][1])*(globalPos[0] - limitReg[0][0])) < 0  
      return into1Point

    elif( dimLimit > 4 ):
      auxArr = []
      for i in range(0,4):
        into1Point = ((limitReg[(i+1)%4][0] - limitReg[i][0])*(globalPos[1] - limitReg[i][1]) - (limitReg[(i+1)%4][1] - limitReg[i][1])*(globalPos[0] - limitReg[i][0])) < 0 
        auxArr.append(int(into1Point))
      
      if( sum(auxArr) == 0 ):
        return True
      else:
        return False

  def addCiclec(self, image, circlex, circley ):
    
    imageC = cv2.circle(image, ( circlex, circley), radius = 3, color=(0, 0, 255), thickness=-1)
    return imageC

  def drawCenter(self, frame):

    validFrame = False

    for i in range(0, self.carLabel.count):

      posX = self.carLabel.arrCentx[i]
      posY = self.carLabel.arrCenty[i]
      validFrame = self.intoRegion( ( posX, posY), self.limitReg1 )
      validFrame = self.intoRegion( ( posX, posY), self.limitReg1 ) and self.intoRegion( ( posX, posY), self.limitReg2 )
      
      if validFrame:
        frame = self.addCiclec( frame, posX, posY )

    for i in range(0, self.motorLabel.count):
      
      posX = self.motorLabel.arrCentx[i]
      posY = self.motorLabel.arrCenty[i]
      validFrame = self.intoRegion( ( posX, posY), self.limitReg1 )
      validFrame = self.intoRegion( ( posX, posY), self.limitReg1 ) and self.intoRegion( ( posX, posY), self.limitReg2 )
    
      if validFrame:
        frame = self.addCiclec( frame, posX, posY)

    for i in range(0, self.busLabel.count):
      
      posX = self.busLabel.arrCentx[i]
      posY = self.busLabel.arrCenty[i]
      validFrame = self.intoRegion( ( posX, posY), self.limitReg1 )
      validFrame = self.intoRegion( ( posX, posY), self.limitReg1 ) and self.intoRegion( ( posX, posY), self.limitReg2 )
      
      if validFrame:  
        frame = self.addCiclec( frame, posX, posY )

    for i in range(0, self.truckLabel.count):
      
      posX = self.truckLabel.arrCentx[i]
      posY = self.truckLabel.arrCenty[i]
      validFrame = self.intoRegion( ( posX, posY), self.limitReg1 )
      validFrame = self.intoRegion( ( posX, posY), self.limitReg1 ) and self.intoRegion( ( posX, posY), self.limitReg2 )
      
      if validFrame:  
        frame = self.addCiclec( frame, posX, posY)

    return frame

  def createDatset(self, idStart, idEnd, pathStart, pathEnd):

    count = 0

    for i in range(idStart,idEnd):
      video = cv2.VideoCapture(pathStart + str(i) + '.mp4')
      
      while(video.isOpened()):
          ret, frame = video.read()
          cv2.imshow('frame', frame)

          if cv2.waitKey(10) == ord('s'):
            count = count + 1
            cv2.imwrite(pathEnd + str(count) + '.jpg' , frame)

          if cv2.waitKey(1) & 0xFF == ord('q'):
              break


      video.release()
      cv2.destroyAllWindows()


  def displayVideo(self, rawData, modelTorch):
    
    while(rawData.isOpened()):
      ret, frame    = rawData.read()
      print('Print Dimension: ', frame.shape)
      frameDetect   = modelTorch(frame)
      print('Result: ', frameDetect.xyxy)
      self.getCenter(frameDetect, labelUsed)
      framemodDetect  = np.squeeze(frameDetect.render())
      framemodCir = self.drawCenter(framemodDetect)

      self.carLabel.destroyObject()
      self.motorLabel.destroyObject()
      self.busLabel.destroyObject()
      self.truckLabel.destroyObject()

      cv2.imshow('Frame With Detection',framemodCir)

      if cv2.waitKey(25) & 0xFF == ord('q'):
          break
            
    rawData.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
  
  pathVideos = 'C:/Users/julit/Downloads/Camara/video_'
  pathDataTrain = 'C:/Users/julit/Proyectos/cameraAA/dataset/train/'
  pathDataValid = 'C:/Users/julit/Proyectos/cameraAA/dataset/validation/'
  
  labelUsed = [ 2, 3, 5, 7 ]
  limitReg1 = [  (624,300), (310,8) ]
  limitReg2 = [ (302,31), (261,26), (117,48), (162,124), (338,97)  ]
  cameraD1  = trafficCamera(labelUsed, limitReg1, limitReg2)
  
  limitReg1 = [  (624,300), (310,8) ]
  limitReg2 = [ (302,31), (261,26), (117,48), (162,124), (338,97)  ]
  cameraD2  = trafficCamera(labelUsed, limitReg1, limitReg2)
  
  limitReg1 = [  (624,300), (310,8) ]
  limitReg2 = [ (302,31), (261,26), (117,48), (162,124), (338,97)  ]
  cameraD3  = trafficCamera(labelUsed, limitReg1, limitReg2)
  
  cameraD1.createDatset(2, 1, pathVideos, pathDataTrain)
  
  model = torch.hub.load('ultralytics/yolov5','yolov5s')
  
  dataCamera1 = cv2.VideoCapture(pathVideos+'11.mp4')
  dataCamera2 = cv2.VideoCapture(pathVideos+'4.mp4')
  dataCamera3 = cv2.VideoCapture(pathVideos+'2.mp4')

  #cameraD1.displayVideo(dataCamera1, model)
  #cameraD2.displayVideo(dataCamera2, model)
  #cameraD3.displayVideo(dataCamera3, model)
  
  while(True):

    ret, frame    = dataCamera1.read()
    frameDetect   = model(frame)
    cameraD1.getCenter(frameDetect, labelUsed)
    framemodDetect  = np.squeeze(frameDetect.render())
    framemodCir1 = cameraD1.drawCenter(framemodDetect)

    ret, frame    = dataCamera2.read()
    frameDetect   = model(frame)
    cameraD2.getCenter(frameDetect, labelUsed)
    framemodDetect  = np.squeeze(frameDetect.render())
    framemodCir2 = cameraD2.drawCenter(framemodDetect)

    ret, frame    = dataCamera1.read()
    frameDetect   = model(frame)
    cameraD3.getCenter(frameDetect, labelUsed)
    framemodDetect  = np.squeeze(frameDetect.render())
    framemodCir3 = cameraD3.drawCenter(framemodDetect)

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

    cv2.imshow('Camero 1',framemodCir1)
    cv2.imshow('Camero 1',framemodCir2)
    cv2.imshow('Camero 1',framemodCir3)

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


