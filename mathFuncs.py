import cv2

def resizeImg(frame, scaleF):
    
    print('Original Dimensions : ',frame.shape)
    width = int(frame.shape[1] * scaleF / 100)
    height = int(frame.shape[0] * scaleF / 100)
    dim = (width, height)

    resizedFrame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    
    return resizedFrame
