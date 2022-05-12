import cv2
import numpy as np
import math

class kernelTrans:
    def __init__(self, thresHold, limitReg,  posRef, scaleF):
        self.color     = (0, 255, 0)
        #self.fgbg      = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=8,detectShadows=False)
        self.fgbg      = cv2.createBackgroundSubtractorKNN(history = 100, dist2Threshold = 200.0, detectShadows=False)
        self.kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1,1))
        self.lon       = 0
        self.w         = 0 
        self.limitReg  = limitReg
        self.thresHold = thresHold
        self.posRef    = posRef
        self.scaleF    = scaleF
    
    def intoRegion(self, globalPos, limitReg ):

        dimLimit = len(limitReg)
        
        if( dimLimit == 2 ):
            into1Point = ((limitReg[1][0] - limitReg[0][0])*(globalPos[1] - limitReg[0][1]) - (limitReg[1][1] - limitReg[0][1])*(globalPos[0] - limitReg[0][0])) < 0  
            return into1Point

        elif( dimLimit >= 3 ):
            auxArr = []
            for i in range(0,dimLimit):
                into1Point = ((limitReg[(i+1)%dimLimit][0] - limitReg[i][0])*(globalPos[1] - limitReg[i][1]) - (limitReg[(i+1)%dimLimit][1] - limitReg[i][1])*(globalPos[0] - limitReg[i][0])) < 0 
                auxArr.append(int(into1Point))
            
            if( sum(auxArr) == dimLimit ):
                return True
            else:
                return False

    def dis2Point(self, x1, y1, x2, y2 ):
        diffx = 1.0*(x1-x2)*(x1-x2)
        diffy = 1.0*(y1-y2)*(y1-y2)
        return math.sqrt(diffx + diffy )*self.scaleF


    def transImage(self, frame):

        #fx 0.5 fy 0.5
        #frame = cv2.resize( frame, None, fx = 0.5, fy = 0.5, interpolation=cv2.INTER_CUBIC)
        #frame = cv2.resize( frame, (320,320), interpolation=cv2.INTER_AREA)
        gray  =  cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        imAux = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
        imAux = cv2.drawContours(imAux, [self.limitReg], -1, (255), -1)
        imageArea = cv2.bitwise_and(gray, gray, mask=imAux)

        fgmask = self.fgbg.apply(imageArea)
        #fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, self.kernel)
        #fgmask = cv2.erode(fgmask, None, iterations=1)
        #fgmask = cv2.dilate(fgmask, None, iterations=1)
        cnts   = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        
        for cnt in cnts:
            areaPixel = cv2.contourArea(cnt)
            
            if areaPixel > self.thresHold[0]:
                cv2.drawContours(frame, cnt , -1, self.color, 1)
                x, y, w, h = cv2.boundingRect(cnt)
                #print('x: ', x, ' y: ', y, ' w: ', w, ' h: ',h)
                centx = x + int(0.5*w)
                centy = y + int(0.5*h) 
                if w > self.thresHold[1]  and h > self.thresHold[2] :
                    
                    if self.intoRegion( (centx, centy), self.limitReg ):
                        cv2.rectangle(frame, (x,y), (x+w, y+h),(0,233,255), 2)
                        #self.lon  = self.lon + h
                        self.lon = self.dis2Point(self.posRef[0], centx, self.posRef[1], centy ) + self.lon 
                        #suma     = '(' + str(w)+',' +str(h)+')'
                        #cv2.putText(frame, str(suma), (centx, centy), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
                        cv2.putText(frame, str(self.lon), (centx, centy), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
        return fgmask 
                
    
    def displayVideo(self, text,frame, fgmask):

        cv2.drawContours(frame, [self.limitReg] , -1, self.color, 2)
        cv2.imshow( text + 'mask' , fgmask)
        cv2.imshow( text , frame)
    
        
    def reset(self):
        self.lon = 0