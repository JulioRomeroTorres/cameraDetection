
from trafficCam import trafficCamera
from controller import plcS7
from mathFuncs import resizeImg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch

refCamera2 = (348 ,355)
scaleCamera2 = 0.08

labelUsed = [ 0, 1 ]

limitReg1 = [  (349,8), (322,8) ]
limitReg2 = [ (587,346), (347,11), (321,10),(273,344)  ]
cameraD2  = trafficCamera( scaleCamera2, refCamera2, labelUsed, limitReg1, limitReg2)

ipPlc   = '192.168.0.21'
rackPlc = 0
slotPlc = 1
plcSem =   plcS7(ipPlc, rackPlc, slotPlc)

model = torch.hub.load('../yolov5','custom', path = 'models/best.pt', source = 'local')

dataCamera1 = cv2.VideoCapture('rtsp://admin:dcsautomation123@192.168.0.101/80')
while( True ):

    ret1, frame   = dataCamera1.read()    
    if ret1:
        
        frame = resizeImg(frame,60)
        frameDetect   = model(frame)
        cameraD2.getCenter(frameDetect, labelUsed)
        framemodDetect  = np.squeeze(frameDetect.render())
        framemodCir1, arrCar, arrTruck = cameraD2.drawCenter(framemodDetect)
        cv2.imshow('Camero 1 Ori', frame)
        cv2.imshow('Camero 1', framemodCir1)
        #plcSem.sendData(np.array(arrTruck).sum())


    cameraD2.carLabel.destroyObject()
    cameraD2.motorLabel.destroyObject()
    cameraD2.busLabel.destroyObject()
    cameraD2.truckLabel.destroyObject()

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
        
dataCamera1.release()
cv2.destroyAllWindows()



