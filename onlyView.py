

import cv2

if __name__ == '__main__':
  
  dataCamera2 = cv2.VideoCapture('rtsp://admin:dcsautomation123@192.168.0.100/80')

  while( True ):

    
    ret2, frame  = dataCamera2.read()
    if ret2:

      cv2.imshow('Camero 2 Ori', frame)


    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
          
  dataCamera2.release()
  cv2.destroyAllWindows()


