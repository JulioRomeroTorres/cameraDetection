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


def getCenter(frameDetect, arrIdlabel):

  global carLabel 
  global motorLabel  
  global busLabel    
  global truckLabel
  global labelUsed 
  
  resultPred = frameDetect.xyxy
  tensorDim = list(resultPred[0].size())

  #print('Result Prediction: ', resultPred[0])
  print('Tensor Dim', tensorDim)
  for i in range(0, tensorDim[0] ):
    print('Raaaaa 1', tensorDim[0])
    if resultPred[0][i,5].item() == arrIdlabel[0]:
      carLabel.defineCenter(resultPred[0][i,0].item(), resultPred[0][i,1].item(), resultPred[0][i,2].item(), resultPred[0][i,3].item() )  
      carLabel.count = carLabel.count + 1
      carLabel.arrPrecis.append(resultPred[0][i,4].item())
      print('Count carr : ', carLabel.count)

    elif resultPred[0][i,5].item() == arrIdlabel[1]: 
      motorLabel.defineCenter(resultPred[0][i,0].item(), resultPred[0][i,1].item(), resultPred[0][i,2].item(), resultPred[0][i,3].item() )
      motorLabel.count = motorLabel.count + 1
      motorLabel.arrPrecis.append(resultPred[0][i,4].item())
      print('Count motor : ', motorLabel.count)

    elif resultPred[0][i,5].item() == arrIdlabel[2]:
      busLabel.defineCenter(resultPred[0][i,0].item(), resultPred[0][i,1].item(), resultPred[0][i,2].item(), resultPred[0][i,3].item() )
      busLabel.count = busLabel.count + 1
      busLabel.arrPrecis.append(resultPred[0][i,4].item())
      print('Count bus : ', busLabel.count)

    elif resultPred[0][i,5].item() == arrIdlabel[3]:
      truckLabel.defineCenter(resultPred[0][i,0].item(), resultPred[0][i,1].item(), resultPred[0][i,2].item(), resultPred[0][i,3].item() )  
      truckLabel.count = truckLabel.count + 1
      truckLabel.arrPrecis.append(resultPred[0][i,4].item())
      print('Count truck : ', truckLabel.count)
    


def intoRegion( globalPos, limitReg ):

  dimLimit = len(limitReg)
  
  if( dimLimit == 2 ):
    into1Point = ((limitReg[1][0] - limitReg[0][0])*(globalPos[1] - limitReg[0][1]) - (limitReg[1][1] - limitReg[0][1])*(globalPos[0] - limitReg[0][0])) > 0  
    return into1Point

  elif( dimLimit == 4 ):
    auxArr = []
    for i in range(0,4):
      into1Point = ((limitReg[(i+1)%4][0] - limitReg[i][0])*(globalPos[1] - limitReg[i][1]) - (limitReg[(i+1)%4][1] - limitReg[i][1])*(globalPos[0] - limitReg[i][0])) > 0 
      auxArr.append(int(into1Point))
    
    if( sum(auxArr) == 0 ):
      return True
    else:
      return False

def addCiclec( image, circlex, circley ):
  
  imageC = cv2.circle(image, ( circlex, circley), radius = 10, color=(0, 0, 255), thickness=-1)
  return imageC

def drawCenter(frame):

  global carLabel 
  global motorLabel  
  global busLabel    
  global truckLabel
  global labelUsed 

  for i in range(0, carLabel.count):
    frame = addCiclec( frame, carLabel.arrCentx[i], carLabel.arrCenty[i] )
  
  for i in range(0, motorLabel.count):
    frame = addCiclec( frame, motorLabel.arrCentx[i], motorLabel.arrCenty[i] )

  for i in range(0, busLabel.count):
    frame = addCiclec( frame, busLabel.arrCentx[i], busLabel.arrCenty[i] )

  for i in range(0, truckLabel.count):
    frame = addCiclec( frame, truckLabel.arrCentx[i], truckLabel.arrCenty[i] )

  return frame

def createDatset(idStart, idEnd, pathStart, pathEnd):

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


def displayVideo(rawData, modelTorch):
  
  global carLabel 
  global motorLabel  
  global busLabel    
  global truckLabel
  global labelUsed 
  
  while(rawData.isOpened()):
    ret, frame    = rawData.read()
    frameDetect   = modelTorch(frame)
    getCenter(frameDetect, labelUsed)
    
    framemodDetect  = np.squeeze(frameDetect.render())
    framemodCir = drawCenter(framemodDetect)

    carLabel.destroyObject()
    motorLabel.destroyObject()
    busLabel.destroyObject()
    truckLabel.destroyObject()

    cv2.imshow('Frame With Detection',framemodCir)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
          
  rawData.release()
  cv2.destroyAllWindows()


if __name__ == '__main__':
  
  pathVideos = 'C:/Users/julit/Downloads/Camara/video_'
  pathDataTrain = 'C:/Users/julit/Proyectos/cameraAA/dataset/train/'
  pathDataValid = 'C:/Users/julit/Proyectos/cameraAA/dataset/validation/'
  
  labelUsed = [ 6, 2, 3, 5, 7 ]

  carLabel    = labelDetected()
  motorLabel  = labelDetected()
  busLabel    = labelDetected()
  truckLabel = labelDetected()

  createDatset(2, 1, pathVideos, pathDataTrain)
  
  model = torch.hub.load('ultralytics/yolov5','yolov5s')
  data = cv2.VideoCapture(pathVideos+'11.mp4')

  displayVideo(data, model)
  
'''model = torch.hub.load('ultralytics/yolov5','yolov5s')

  frame = cv2.imread('C:/Users/julit/Downloads/Camara/truckTest.png')
  frameDetect   = model(frame)
  getCenter(frameDetect, labelUsed)
  
  framemodDetect  = np.squeeze(frameDetect.render())
  framemodCir = drawCenter(framemodDetect)

  carLabel.destroyObject()
  motorLabel.destroyObject()
  busLabel.destroyObject()
  truckLabel.destroyObject()

  cv2.imshow('Frame With Detection',framemodCir)
  cv2.waitKey(0) 
    
  #closing all open windows 
  cv2.destroyAllWindows()''' 


