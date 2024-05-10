import cv2
import face_recognition
import mediapipe as mp
from PIL import Image, ImageTk
import customtkinter

#TODO: Set up tkinter to make an actual program for this class to be used in.

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("dark-blue")

root = customtkinter.CTk()
root.maxsize(1020, 740)
root.minsize(1020, 740)

#------------------------------------------------------------------
#TOP FRAME START
#------------------------------------------------------------------
top_frame = customtkinter.CTkFrame(root, height=100, width=1000)
top_frame.grid(row=0, column=0, padx=(10, 10), pady=5, sticky='EW')
top_frame.grid_propagate(False)


top_label = customtkinter.CTkLabel(master=top_frame, text="Face Recoginition Software", font=('Helvetica', 30), bg_color="green")
top_label.place(relx=.5, rely=.5, anchor="center")
#------------------------------------------------------------------
#TOP FRAME END
#------------------------------------------------------------------

#------------------------------------------------------------------
#MAIN FRAME START
#------------------------------------------------------------------
main_frame = customtkinter.CTkFrame(root, height=400, width=1000)
main_frame.grid(row=1, column=0, padx=10, pady=5, sticky='EW')
main_frame.grid_propagate(False)

left_cap_label = customtkinter.CTkLabel(master=main_frame, text="left_cap_label")
left_cap_label.grid(row=0, column=0)

right_cap_label = customtkinter.CTkLabel(master=main_frame, text="right_cap_label")
right_cap_label.grid(row=0, column=1)
#------------------------------------------------------------------
#MAIN FRAME END
#------------------------------------------------------------------

#------------------------------------------------------------------
#BOTTOM FRAME START
#------------------------------------------------------------------
bottom_frame = customtkinter.CTkFrame(root, height=200, width=1000)
bottom_frame.grid(row=2, column=0, padx=10, pady=5, sticky='EW')
bottom_frame.grid_propagate(False)

bottom_button = customtkinter.CTkButton(master=bottom_frame, width=280, height=56, text="Start Camera", font=('Helvetica', 30))
bottom_button.place(relx=.5, rely=.5, anchor="center")
#------------------------------------------------------------------
#BOTTOM FRAME END
#------------------------------------------------------------------

#label = customtkinter.CTkLabel(master=frame, text="Face Detection")
#label.pack(pady=12, padx=10)

class FaceDetector():

    face_match = False
    inital_image_name = None
    face_saved = None

    def __init__(self, minDetectionCon = 0.5):

        self.minDetectionCon = minDetectionCon

            
    def save_face(self, frame, fileName):
        imgArr = mp.ImageFrame(image_format= mp.ImageFormat.SRGB, data=frame).numpy_view()
        newImg = Image.fromarray(imgArr)
        newImg.save(fileName)
        self.inital_image_name = fileName
        self.face_saved = True

    def compare_faces(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        unknown_face_image = face_recognition.load_image_file("2.png")
        unknown_face_encoding = face_recognition.face_encodings(unknown_face_image)[0]

        known_face_image = face_recognition.load_image_file(self.inital_image_name)
        known_face_encoding = face_recognition.face_encodings(known_face_image)

        if True in face_recognition.compare_faces(known_face_encoding, unknown_face_encoding):
            self.face_match = True
        else:
            self.face_match = False

    def find_faces(self, frame, draw = True):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(img_rgb)
        face_encodings = face_recognition.face_encodings(img_rgb, face_locations)

        for(top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            if self.face_saved != True:
                self.save_face(frame, "1.png")
            else:
                self.save_face(frame, "2.png")
                self.compare_faces(frame)
                print(self.face_match)

            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 255), 2)

        return frame
    
    def show_frames():

        #Getting your video capture, capture 0 is default.
        cap = cv2.VideoCapture(0)
        detector = FaceDetector(0.75)

        #Get latest frame and convert into image
        cv2image= cv2.cvtColor(cap.read()[1],cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)

        #Convert image to PhotoImage
        imgtk = ImageTk.PhotoImage(image = img)
        label.imgtk = imgtk
        label.configure(image = imgtk)

#def main():
#    #Naming window, and allowing fullscreen.
#    cv2.namedWindow("Face Detection", cv2.WND_PROP_FULLSCREEN)
#
#    #Getting your video capture, capture 0 is default.
#    cap = cv2.VideoCapture(0)
#
#    detector = FaceDetector(0.75)
#
#    while True:
#        #Reading in the capture image
#        success, frame = cap.read()
#
#        #Finding the faces in the capture, and then displaying them.
#        frame = detector.find_faces(frame)
#
#        cv2.imshow("Face Detection", frame)
#
#        #Controlling the FPS
#        key = cv2.waitKey(20)
#        #Exit on ESC
#        if key == 27:
#            break
#
#    #Releases capture object
#    cap.release()
#    #Remove window
#    cv2.destroyWindow("Preview")


#if __name__ == "__main__":
    #main()

root.mainloop()