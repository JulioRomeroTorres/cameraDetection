
from trafficCam import trafficCamera
from controller import plcS7
from mathFuncs import resizeImg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch

refCamera3 = (353, 352)
scaleCamera3 = 0.08

labelUsed = [ 0, 1]

limitReg1 = [  (371,12), (339,9) ]
limitReg2 = [ (567,352), (375,18), (338,15), (268,348) ]
cameraD3  = trafficCamera( scaleCamera3, refCamera3, labelUsed, limitReg1, limitReg2)

ipPlc   = '192.168.0.21'
rackPlc = 0
slotPlc = 1
plcSem =   plcS7(ipPlc, rackPlc, slotPlc)

model = torch.hub.load('../yolov5','custom', path = 'models/best.pt', source = 'local')

dataCamera1 = cv2.VideoCapture('rtsp://admin:dcsautomation123@192.168.0.102/80')
while( True ):

    ret1, frame   = dataCamera1.read()    
    if ret1:
        
        frame = resizeImg(frame,60)
        frameDetect   = model(frame)
        cameraD3.getCenter(frameDetect, labelUsed)
        framemodDetect  = np.squeeze(frameDetect.render())
        framemodCir1, arrCar, arrTruck = cameraD3.drawCenter(framemodDetect)
        cv2.imshow('Camero 1 Ori', frame)
        cv2.imshow('Camero 1', framemodCir1)
        #plcSem.sendData(np.array(arrTruck).sum())


    cameraD3.carLabel.destroyObject()
    cameraD3.motorLabel.destroyObject()
    cameraD3.busLabel.destroyObject()
    cameraD3.truckLabel.destroyObject()

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
        
dataCamera1.release()
cv2.destroyAllWindows()



