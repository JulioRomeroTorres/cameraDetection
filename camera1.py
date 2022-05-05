
from trafficCam import trafficCamera
from controller import plcS7
from mathFuncs import resizeImg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch


refCamera1 = (142 ,188)
scaleCamera1 = 0.08

labelUsed = [ 0, 1 ]
limitReg1 = [ (539 ,233), (380 ,82) ]
limitReg2 = [ (481 ,200), (366 ,84), (378 ,105), (376 ,137), (337 ,164), (281 ,180), (225 ,184), (170 ,182), (154 ,265), (226 ,269), (322 ,262), (393 ,243) , (476 ,194) ]
cameraD1  = trafficCamera( scaleCamera1, refCamera1, labelUsed, limitReg1, limitReg2)

ipPlc   = '192.168.0.21'
rackPlc = 0
slotPlc = 1
plcSem =   plcS7(ipPlc, rackPlc, slotPlc)

model = torch.hub.load('../yolov5','custom', path = 'models/best.pt', source = 'local')

dataCamera1 = cv2.VideoCapture('rtsp://admin:dcsautomation123@192.168.0.100/80')
while( True ):

    ret1, frame   = dataCamera1.read()    
    if ret1:
        
        frame = resizeImg(frame,60)
        frameDetect   = model(frame)
        cameraD1.getCenter(frameDetect, labelUsed)
        framemodDetect  = np.squeeze(frameDetect.render())
        framemodCir1, arrCar, arrTruck = cameraD1.drawCenter(framemodDetect)
        cv2.imshow('Camero 1 Ori', frame)
        #cv2.imshow('Camero 1', framemodCir1)
        #plcSem.sendData(np.array(arrTruck).sum())


    cameraD1.carLabel.destroyObject()
    cameraD1.motorLabel.destroyObject()
    cameraD1.busLabel.destroyObject()
    cameraD1.truckLabel.destroyObject()

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
        
dataCamera1.release()
cv2.destroyAllWindows()



