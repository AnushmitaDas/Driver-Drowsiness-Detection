import cv2
import dlib
import numpy as np
from math import sqrt
from playsound import playsound
from threading import Thread

#video capture through webcam
cap= cv2.VideoCapture(0)

#Dlib detector object made to detect frontal face
detector= dlib.get_frontal_face_detector()

#Dlib predictor object to predict 68 landmarks on the face
predictor= dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#variable for counting frames
count=0 

#This thread runs to play alarm on detecting drowsiness
def playy():
    playsound("sound.wav")

#returns eucledian distance between two points 
def eucledian_dist(p1,p2):

    x1,x2=p1.x,p2.x
    y1,y2=p1.y,p2.y
    res= sqrt(((x2-x1)**2)+ ((y2-y1)**2))

    return res


#returns eye aspect ratio (EAR) 
def aspect_ratio(arr, landmarks):

    A=eucledian_dist(landmarks.part(arr[1]),landmarks.part(arr[5]))
    B=eucledian_dist(landmarks.part(arr[2]),landmarks.part(arr[4]))
    C=eucledian_dist(landmarks.part(arr[0]),landmarks.part(arr[3]))

    ear=(A+B)/(2.0*C)

    return ear


#constantly captures frames from webcam until "q" is pressed on the keyboard
while True:

    _,img= cap.read()

    #converts from BGR to gray image to reduce the computational time of processing an image
    gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #using the detector object to detect face from the img frames captured
    faces= detector(gray)

    for face in faces:
        x1,y1=face.left(),face.top()
        x2,y2=face.right(),face.bottom()
        
        #converts the detected face to 68 landmarks or (x,y) points
        landmarks= predictor(gray, face)


        left= aspect_ratio([36,37,38,39,40,41],landmarks)       #left EAR --- arguments include the eye landmarks
        right= aspect_ratio([42,43,44,45,46,47],landmarks)      #rightEAR 

        ear = (left+right)/2.0

        cv2.putText(img, "Eye Aspect Ratio: {:.2f}".format(ear), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        EAR_Thres=0.3           #EAR threshold below which if EAR goes then blink is detected
        EAR_Dur_Thres=48        #Threshold of the count of frames for which the eye remains below threshold

        if(left < EAR_Thres and right < EAR_Thres):
            count+=1

            if(count>=EAR_Dur_Thres):                       
                T = Thread(target=playy) # create thread 
                T.start() # Launch created thread that rings because drowsiness detected
                cv2.putText(img, "DROWSY", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        else:
            count = 0
            cv2.putText(img, "NOT DROWSY", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        print(left, end=" ")
        print(right)
        

    cv2.imshow("output", img)

    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()