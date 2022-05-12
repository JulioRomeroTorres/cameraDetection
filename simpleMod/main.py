from controller import plcS7
from cameraIp import kernelTrans

import cv2
import numpy as np
import math

import snap7
from snap7.common import check_error, load_library, ipv4
from snap7.exceptions import Snap7Exception


if __name__ == '__main__':

    ipPlc = '192.168.0.120'
    rackPlc = 0
    slotPlc = 1

    plcControl = plcS7(ipPlc, rackPlc, slotPlc)

    '''pathVideo = 'rtsp://admin:dcsautomation123@192.168.0.'
    videoCam1 = cv2.VideoCapture(pathVideo + '100/80' )
    videoCam2 = cv2.VideoCapture(pathVideo + '101/80' )
    videoCam3 = cv2.VideoCapture(pathVideo + '102/80' )'''

    pathVideo = 'C:/Users/julit/Downloads/Camara/video'
    videoCam1 = cv2.VideoCapture('C:/Users/julit/Downloads/Camara/videoCam1/test3.mp4' )
    videoCam2 = cv2.VideoCapture('C:/Users/julit/Downloads/Camara/videoCam2/test3.mp4' )
    videoCam3 = cv2.VideoCapture('C:/Users/julit/Downloads/Camara/videoCam3/test5.mp4')
    
    
    posRef1 = ( 352, 354 )
    posRef2 = ( 132, 196 )
    posRef3 = ( 342, 357 )

    limitReg1 = np.array([ [443, 351], [330, 6], [318, 5], [262, 348] ])
    limitReg2 = np.array([ [210 ,272], [376 ,259], [438 ,244], [482 ,195], [420 ,126], [365 ,83], [205 ,176] ])
    limitReg3 = np.array([ [432, 327], [350, 13], [338, 9], [268, 316] ])

    thLimit1 = [2100, 50, 70]
    thLimit2 = [3000, 80, 80]
    thLimit3 = [1000, 50, 50]

    scaleF1 = 0.08
    scaleF2 = 0.08
    scaleF3 = 0.08

    kernelT1 = kernelTrans(thLimit1, limitReg1, posRef1, scaleF1)
    kernelT2 = kernelTrans(thLimit2, limitReg2, posRef2, scaleF2)
    kernelT3 = kernelTrans(thLimit3, limitReg3, posRef3, scaleF3)

    while True:
        try:
            
            plcControl.HoB = 1 - plcControl.HoB
            '''ret1, frame = videoCam1.read()
            if ret1:    
    
                maskImage = kernelT1.transImage(frame)
                #plcControl.sendData(kernelT1.lon)
                kernelT1.displayVideo( 'frame 1', frame, maskImage)

             
            ret2, frame = videoCam2.read()
            if ret2:

                maskImage = kernelT2.transImage(frame)
                #plcControl.sendData(kernelT2.lon) 
                kernelT2.displayVideo('frame 2', frame, maskImage)
            '''
            ret3, frame = videoCam3.read()
            if ret3:

                maskImage = kernelT3.transImage(frame)
                #plcControl.sendData(kernelT3.lon)
                kernelT3.displayVideo('frame 3', frame, maskImage)
            
            if cv2.waitKey(5) & 0xFF == ord ('q'):
                break
            
            kernelT1.reset()
            kernelT2.reset()
            kernelT3.reset()

        except Snap7Exception as e:
            try:
                print("Error with PLC")
            except Snap7Exception as e:
                print("B")
                continue
    
        
    videoCam1.release()
    videoCam2.release()
    videoCam3.release()
    cv2.destroyAllWindows()
