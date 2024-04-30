import cv2
import mediapipe as mp

cv2.namedWindow("Preview")
cap = cv2.VideoCapture(0)

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.75)

while True:
    #Reading in the capture image and displaying it
    success, frame = cap.read()

    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)

    if results.detections:
        for id, detection in enumerate(results.detections):
            #mpDraw.draw_detection(frame, detection)

            #Creating the rectangle for drawing
            bboxC = detection.location_data.relative_bounding_box
            imgH, imgW, imgC = frame.shape
            bbox = int(bboxC.xmin * imgW), int(bboxC.ymin * imgH), \
                   int(bboxC.width * imgW), int(bboxC.height * imgH)
            
            #color = (255, 0, 255)
            
            #if detection.score >= 90:
                #color = (0, 255, 0)
            #else:
                #color = (255, 0, 255)

            cv2.rectangle(frame, bbox, (255, 0, 255), 2)
            cv2.putText(frame, f'{int(detection.score[0]*100)}%', 
                        (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_PLAIN,
                        2, (255, 0, 255), 2)

    cv2.imshow("Preview", frame)

    #Controlling the FPS
    key = cv2.waitKey(20)

    #Exit on ESC
    if key == 27:
        break

#Releases capture object
cap.release()
#Remove window
cv2.destroyWindow("Preview")