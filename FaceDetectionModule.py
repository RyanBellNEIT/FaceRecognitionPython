import cv2
import face_recognition
import mediapipe as mp
from PIL import Image, ImageTk
import customtkinter
import os

#TODO: Fix lag when start camera button is pressed.

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


top_label = customtkinter.CTkLabel(master=top_frame, text="Face Recoginition", font=('Helvetica', 30))
top_label.place(relx=.5, rely=.5, anchor="center")
#------------------------------------------------------------------
#TOP FRAME END
#------------------------------------------------------------------

#------------------------------------------------------------------
#MAIN FRAME START
#------------------------------------------------------------------
main_frame = customtkinter.CTkFrame(root, height=400, width=1000)
main_frame.grid(row=1, column=0, padx=10, pady=5, sticky='EW')

left_cap_label = customtkinter.CTkLabel(master=main_frame, padx=5, text="")
left_cap_label.place(relx=.25, rely=.5, anchor="center")

right_cap_label = customtkinter.CTkLabel(master=main_frame, padx=5, text="")
right_cap_label.place(relx=.75, rely=.5, anchor="center")
#------------------------------------------------------------------
#MAIN FRAME END
#------------------------------------------------------------------

def reset_match(face_detector):
    #Resetting top label
    top_label.configure(text="Face Recoginition", font=('Helvetica', 30), bg_color="green")
    top_label.update()

    #Resetting left picture
    left_cap_label.imgtk = None
    left_cap_label.configure(image=None)
    left_cap_label.update()

    #Resetting right picture
    right_cap_label.imgtk = None
    right_cap_label.configure(image=None)
    right_cap_label.update()

    #Resetting right picture
    bottom_button.configure(text="Start Camera", command=start_camera)
    bottom_button.update()

    face_detector.face_saved[0], face_detector.face_saved[1] = False, False
    

def update_cam(frame, is_not_first_pic):
        if is_not_first_pic != True:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            left_cap_label.imgtk = imgtk
            left_cap_label.configure(image=imgtk)
        else:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            right_cap_label.imgtk = imgtk
            right_cap_label.configure(image=imgtk)


def start_camera():
    bottom_button.configure(state="disabled", text="Capturing...")
    bottom_button.update()

    #Allows for multi-platform use, using CAP_DSHOW on windows make camera work faster, but doesn't work on other platforms.
    if os.name == 'nt':
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(0)
    detector = FaceDetector()

    #Second loop of the program, only loops after the first face picture was captured
    while detector.face_saved[0] == True and detector.face_saved[1] != True:
        frame = cap.read()[1]

        #Allows capture to display on tkinter UI
        update_cam(frame, detector.face_saved[0])

        #Need to make it so it uses the class functions and variables, from this function.
        frame = detector.find_faces(frame)
        #Updates on UI after changing frame variable
        update_cam(frame, True)
        
        if detector.face_saved[1] == True:
            top_label.configure(text=" MATCH " if detector.face_match else " NON-MATCH ",
                                bg_color="green" if detector.face_match else "red")
            top_label.update()
            break

        #Controlling the FPS
        key = cv2.waitKey(20)
        root.update()
    
    #First loop of the program, loops when user selects start camera button for the first time.
    while detector.face_saved[0] != True:
        frame = cap.read()[1]

        #Allows capture to display on tkinter UI
        update_cam(frame, detector.face_saved[0])

        #Need to make it so it uses the class functions and variables, from this function.
        frame = detector.find_faces(frame)
        update_cam(frame, False)

        if detector.face_saved[0] == True:
            break

        #Controlling the FPS
        #key = cv2.waitKey(20)
        root.update()

    
    if detector.face_saved[1] == True:
        bottom_button.configure(state="normal", text="Clear", command= lambda: reset_match(detector))
        bottom_button.update()
    else:
        bottom_button.configure(state="normal", text="Start Camera", command=start_camera)
        bottom_button.update()
    cap.release()

def setup_start():
    bottom_button.configure(text="Initializing...")
    bottom_button.update()
    start_camera()


#------------------------------------------------------------------
#BOTTOM FRAME START
#------------------------------------------------------------------
bottom_frame = customtkinter.CTkFrame(root, height=200, width=1000)
bottom_frame.grid(row=2, column=0, padx=10, pady=5, sticky='EW')
bottom_frame.grid_propagate(False)

bottom_button = customtkinter.CTkButton(master=bottom_frame, width=280, height=56, text="Start Camera", font=('Helvetica', 30), command=setup_start)
bottom_button.place(relx=.5, rely=.5, anchor="center")
#------------------------------------------------------------------
#BOTTOM FRAME END
#------------------------------------------------------------------

class FaceDetector():

    face_saved = [False, False]
    face_match = False
    inital_image_name = None

    def __init__(self, minDetectionCon = 0.5):
        self.minDetectionCon = minDetectionCon

            
    def save_face(self, frame, fileName):
        imgArr = mp.ImageFrame(image_format= mp.ImageFormat.SRGB, data=frame).numpy_view()
        newImg = Image.fromarray(imgArr)
        newImg.save(fileName)
        self.inital_image_name = fileName
        if self.face_saved[0] != True:
            self.face_saved[0] = True
        else:
            self.face_saved[1] = True


    def compare_faces(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        unknown_face_image = face_recognition.load_image_file("2.png")
        unknown_face_encoding = face_recognition.face_encodings(unknown_face_image)[0]

        known_face_image = face_recognition.load_image_file("1.png")
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
            if self.face_saved[0] != True:
                self.save_face(frame, "1.png")
            elif self.face_saved[1] != True:
                self.save_face(frame, "2.png")
                self.compare_faces(frame)

            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 255), 2)

        return frame

root.mainloop()