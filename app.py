# import the necessary packages
import streamlit as st
from streamlit_webrtc import webrtc_streamer
from scipy.spatial import distance as dist
from imutils import face_utils
import face_recognition
import pandas as pd
import numpy as np
from datetime import datetime
import os
import av
import cv2
import dlib
import imutils
import pickle

st.title("Web Application Demo")

print("[INFO] loading encodings...")
data = pickle.loads(open("encodings.pickle", "rb").read())
print("[INFO] encodings loaded")

def Attendance(name):
    with open('Attendance_Registry.csv','r+') as f:
        DataList = f.readlines()
        names = []
        for dtl in DataList:
            ent = dtl.split(',')
            names.append(ent[0])
        if name not in names:
            curr = datetime.now()
            dt = curr.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dt}')

class VideoProcessor1:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        rgb_resized = cv2.resize(img,(0,0),None,0.25,0.25) 
        rgb_resized = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodeKnown = list(data["encodings"])

        facesInFrame = face_recognition.face_locations(rgb_resized)
        encodesInFrame = face_recognition.face_encodings(rgb_resized,facesInFrame)

        for encFace,faceLoc in zip(encodesInFrame,facesInFrame):
                matchList = face_recognition.compare_faces(encodeKnown,encFace)
                faceDist = face_recognition.face_distance(encodeKnown,encFace)
                match = np.argmin(faceDist)
                if matchList[match]:
                    st_name = data["names"][match]
                    Attendance(st_name)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb = imutils.resize(img, width=750)
        r = img.shape[1] / float(rgb_resized.shape[1])
        # detect the (x, y)-coordinates of the bounding boxes
        # corresponding to each face in the input frame, then compute
        # the facial embeddings for each face
        boxes = face_recognition.face_locations(rgb_resized,
            model="hog")
        encodings = face_recognition.face_encodings(rgb_resized, boxes)
        names = []

        # loop over the facial embeddings
        for encoding in encodings:
            # attempt to match each face in the input image to our known
            # encodings
            matches = face_recognition.compare_faces(data["encodings"],encoding)
            name = "Unknown"
            # check to see if we have found a match
            if True in matches:
                # find the indexes of all matched faces then initialize a
                # dictionary to count the total number of times each face
                # was matched
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                # loop over the matched indexes and maintain a count for
                # each recognized face face
                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1
                # determine the recognized face with the largest number
                # of votes (note: in the event of an unlikely tie Python
                # will select first entry in the dictionary)
                name = max(counts, key=counts.get)
            
            # update the list of names
            names.append(name)

        # loop over the recognized faces
        for ((top, right, bottom, left), name) in zip(boxes, names):
            # rescale the face coordinates
            top = int(top * r)
            right = int(right * r)
            bottom = int(bottom * r)
            left = int(left * r)
            # draw the predicted face name on the image
            loc = cv2.rectangle(img, (left, top), (right, bottom),
                (0, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            img = cv2.putText(loc, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")




def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])
	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)
	# return the eye aspect ratio
	return ear

# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 48
# initialize the frame counter 
COUNTER = 0

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

class VideoProcessor2:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb = imutils.resize(img, width=450)
       
        # detect faces in the grayscale frame
        rects = detector(rgb, 0)
        global COUNTER
        # loop over the face detections
        for rect in rects:
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(rgb, rect)
            shape = face_utils.shape_to_np(shape)
            # extract the left and right eye coordinates, then use the
            # coordinates to compute the eye aspect ratio for both eyes
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            # average the eye aspect ratio together for both eyes
            ear = (leftEAR + rightEAR) / 2.0

            # compute the convex hull for the left and right eye, then
            # visualize each of the eyes
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            # cv2.drawContours(rgb, [leftEyeHull], -1, (0, 255, 0), 1)
            # cv2.drawContours(rgb, [rightEyeHull], -1, (0, 255, 0), 1)

            # check to see if the eye aspect ratio is below the blink
            # threshold, and if so, increment the blink frame counter
            if ear < EYE_AR_THRESH:
                COUNTER += 1
                # if the eyes were closed for a sufficient number of
                # then set off the alert
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    # draw an alert on the frame
                    img = cv2.putText(rgb, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                COUNTER = 0
            
        return av.VideoFrame.from_ndarray(img, format="bgr24")




def main():
    
    activities = ["Home","Face Recognition","Drowsiness Detection","Attendance Registry","About"]
    choice = st.sidebar.selectbox("Select",activities)
    
    if choice == "Home":
        html_temp = """
        <body style="background-color:red;">
        <div style="background-color:teal ;padding:10px">
        <h2 style="color:white;text-align:center;">Home</h2>
        </div>
        </body>
        """
        st.markdown(html_temp, unsafe_allow_html=True)
        st.write('''

        Welcome,

        This app is developed to enhance online education by automatically recognizing the student face, mark the attendance, log the individual's session time and put a drowsiness alert.

        It has three functionalities, 
        
        1. To recognize faces given a few images of the students.
        
        2. Log the student name in the attendance registry along with logged in time.  

        3. Set off an alert whenever drowsiness is detected in the students face. 

        Click on the drop box to choose the functionality.
        
        ''')

    elif choice == "Face Recognition":
        html_temp = """
        <body style="background-color:red;">
        <div style="background-color:teal ;padding:10px">
        <h2 style="color:white;text-align:center;">Face Recognition</h2>
        </div>
        </body>
        """
        st.markdown(html_temp, unsafe_allow_html=True)
        st.write('''
                  
        1. Click the Start button to open webcam.
        
        2. It will automatically recognize the face.
        
        3. Click the Stop button to close the webcam.

        Note: It will output "Unknown" if your image is not present in the dataset directory. 
                
		                                            ''')
        webrtc_streamer(key="FaceRecognition", video_processor_factory=VideoProcessor1)

    elif choice == "Drowsiness Detection":
        html_temp = """
        <body style="background-color:red;">
        <div style="background-color:teal ;padding:10px">
        <h2 style="color:white;text-align:center;">Drowsiness Detection</h2>
        </div>
        </body>
        """
        st.markdown(html_temp, unsafe_allow_html=True)
        st.write('''
        
        1. Click the Start button to open webcam.
        
        2. It will automatically detect the drowsiness and set off an alert.
        
        3. Click the Stop button to close the webcam.

        The alert will be set off if your eyes are partially or fully closed for more than 3 seconds"
                  
                                                     ''')
        webrtc_streamer(key="DrowsinessDetection", video_processor_factory=VideoProcessor2)
    
    elif choice == "Attendance Registry":
        html_temp = """
        <body style="background-color:red;">
        <div style="background-color:teal ;padding:10px">
        <h2 style="color:white;text-align:center;">Attendance Registry</h2>
        </div>
        </body>
        """
        df = pd.read_csv("Attendance_Registry.csv")
        st.write(df)

    elif choice == "About":
        html_temp = """
        <body style="background-color:red;">
        <div style="background-color:teal ;padding:10px">
        <h2 style="color:white;text-align:center;">About</h2>
        </div>
        </body>
        """  
        st.markdown(html_temp, unsafe_allow_html=True)
        st.write('''

        Face Recognition

        This is a few shot learning live face attendance systems. The model will do real-time identification of students in the live class based on few images of students.
        

        Drowsiness detector

        This will detect facial landmarks and extract the eye regions. The algorithm will identify students who are not attentive and drowsing in the class.
        ''')

if __name__ == "__main__":
    main()