import numpy as np
import cv2
import torch

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
  
  while(rawData.isOpened()):
    ret, frame = data.read()
    frameDetect = modelTorch(frame)
    cv2.imshow('Frame With Detection',np.squeeze(frameDetect.render()))

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
          
  data.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  
  pathVideos = 'C:/Users/julit/Downloads/Camara/video_'
  pathDataTrain = 'C:/Users/julit/Proyectos/cameraAA/dataset/train/'
  pathDataValid = 'C:/Users/julit/Proyectos/cameraAA/dataset/validation/'
  
  createDatset(2, 1, pathVideos, pathDataTrain)
  
  model = torch.hub.load('ultralytics/yolov5','yolov5s')
  data = cv2.VideoCapture(pathVideos+'2.mp4')
  displayVideo(data, model)
  print('Raa')
