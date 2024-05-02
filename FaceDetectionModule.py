import cv2
import mediapipe as mp

class FaceDetector():
    def __init__(self, minDetectionCon = 0.5):

        self.minDetectionCon = minDetectionCon

        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)

    def findFaces(self, frame, draw = True):
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)

        bboxs = []

        if self.results.detections:
            for id, detection in enumerate(self.results.detections):

                #Creating the rectangle for drawing
                bboxC = detection.location_data.relative_bounding_box
                imgH, imgW, imgC = frame.shape
                bbox = int(bboxC.xmin * imgW), int(bboxC.ymin * imgH), \
                       int(bboxC.width * imgW), int(bboxC.height * imgH)
                
                bboxs.append([bbox, detection.score])

                if draw:
                    cv2.rectangle(frame, bbox, (255, 0, 255), 2)
                    cv2.putText(frame, f'{int(detection.score[0]*100)}%', 
                        (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_PLAIN,
                        2, (255, 0, 255), 2)
                    

                print(type(detection.score))

                #if detection.score > 0.95:
                    #print("Good image")
                #else:
                    #print("Bad image")
                
        return frame, bboxs

def main():
    #Naming window, and allowing fullscreen.
    cv2.namedWindow("Face Detection", cv2.WND_PROP_FULLSCREEN)

    #Getting your video capture, capture 0 is default.
    cap = cv2.VideoCapture(0)

    detector = FaceDetector(0.75)

    while True:
        #Reading in the capture image
        success, frame = cap.read()

        #Finding the faces in the capture, and then displaying them.
        frame, bboxs = detector.findFaces(frame)

        cv2.imshow("Face Detection", frame)

        #Controlling the FPS
        key = cv2.waitKey(20)
        #Exit on ESC
        if key == 27:
            break

    #Releases capture object
    cap.release()
    #Remove window
    cv2.destroyWindow("Preview")


if __name__ == "__main__":
    main()