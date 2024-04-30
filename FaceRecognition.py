import cv2

cv2.namedWindow("Preview")
vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    cv2.imshow("Preview", frame)
    rval, frame = vc.read()
    
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break

#Releases capture object
vc.release()
#Remove window
cv2.destroyWindow("Preview")